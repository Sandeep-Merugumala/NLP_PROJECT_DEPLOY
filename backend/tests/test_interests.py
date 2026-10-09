"""
Unit tests for interest clustering logic.
Tests the _compute_interests helper logic independently of Flask app context.
Run: python -m pytest backend/tests/test_interests.py -v
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------- Standalone reimplementation for testing (mirrors app.py) ----------
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_interests(papers_db, ratings_db, tfidf_vectorizer, tfidf_matrix, paper_index_map):
    """Standalone version of _compute_interests for unit-testing."""
    upvoted = [
        p for p in papers_db
        if p["id"] in ratings_db and ratings_db[p["id"]] == 1 and p.get("embedding")
    ]
    if len(upvoted) < 3:
        return []

    n_clusters = min(3, len(upvoted))
    X = np.array([p["embedding"] for p in upvoted])

    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    except Exception:
        return []

    feature_names = tfidf_vectorizer.get_feature_names_out()
    interests = []
    for ci in range(n_clusters):
        cluster_papers = [upvoted[i] for i, lbl in enumerate(labels) if lbl == ci]
        if not cluster_papers:
            continue
        indices = [paper_index_map[p["id"]] for p in cluster_papers if p["id"] in paper_index_map]
        if not indices:
            continue
        avg_vec = tfidf_matrix[indices].mean(axis=0)
        top_term_indices = np.argsort(avg_vec)[::-1][:4]
        top_terms = [feature_names[i] for i in top_term_indices]
        interests.append({
            "cluster": ci,
            "label": f"Topic {ci + 1}: {', '.join(top_terms)}",
            "top_terms": top_terms,
            "count": len(cluster_papers),
            "representative": cluster_papers[0]["title"]
        })
    return interests


# ---------- Test helpers ----------

def make_paper(pid, title, abstract, rating=None, dim=16):
    emb = np.random.default_rng(hash(pid) % (2**31)).random(dim).tolist()
    return {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "embedding": emb
    }


def make_tfidf(papers):
    corpus = [f"{p['title']} {p['abstract']}" for p in papers]
    vec = TfidfVectorizer(max_features=16, stop_words="english")
    mat = vec.fit_transform(corpus).toarray()
    index_map = {p["id"]: i for i, p in enumerate(papers)}
    return vec, mat, index_map


# ---------- Tests ----------

def test_returns_empty_with_fewer_than_3_upvotes():
    """Should return [] when upvoted count < 3."""
    papers = [make_paper(f"p{i}", f"Title {i}", f"Abstract text about topic {i}") for i in range(10)]
    ratings = {"p0": 1, "p1": 1}  # only 2 upvotes
    vec, mat, idx_map = make_tfidf(papers)
    result = compute_interests(papers, ratings, vec, mat, idx_map)
    assert result == [], f"Expected empty list, got {result}"


def test_returns_clusters_with_enough_upvotes():
    """Should return ≥1 cluster when ≥3 papers are upvoted."""
    papers = [make_paper(f"p{i}", f"Transformer Model {i}", f"We propose a novel transformer for NLP task {i}") for i in range(10)]
    ratings = {f"p{i}": 1 for i in range(5)}  # 5 upvotes
    vec, mat, idx_map = make_tfidf(papers)
    result = compute_interests(papers, ratings, vec, mat, idx_map)
    assert len(result) >= 1, "Expected at least 1 interest cluster"


def test_cluster_count_bounded_by_3():
    """Cluster count should never exceed 3 regardless of upvote count."""
    papers = [make_paper(f"p{i}", f"Paper {i}", f"Some text about machine learning topic number {i}") for i in range(20)]
    ratings = {f"p{i}": 1 for i in range(20)}
    vec, mat, idx_map = make_tfidf(papers)
    result = compute_interests(papers, ratings, vec, mat, idx_map)
    assert len(result) <= 3, f"Expected ≤3 clusters, got {len(result)}"


def test_each_interest_has_required_fields():
    """Every interest dict should have cluster, label, top_terms, count, representative."""
    papers = [make_paper(f"p{i}", f"Deep Learning {i}", f"Neural networks for vision task {i}") for i in range(10)]
    ratings = {f"p{i}": 1 for i in range(6)}
    vec, mat, idx_map = make_tfidf(papers)
    result = compute_interests(papers, ratings, vec, mat, idx_map)
    for interest in result:
        assert "cluster" in interest
        assert "label" in interest
        assert "top_terms" in interest
        assert "count" in interest
        assert "representative" in interest
        assert len(interest["top_terms"]) > 0


def test_no_crash_with_exact_3_upvotes():
    """Edge case: exactly 3 upvotes should not crash."""
    papers = [make_paper(f"p{i}", f"Title {i}", f"Abstract {i}") for i in range(5)]
    ratings = {"p0": 1, "p1": 1, "p2": 1}
    vec, mat, idx_map = make_tfidf(papers)
    result = compute_interests(papers, ratings, vec, mat, idx_map)  # should not raise
    assert isinstance(result, list)


if __name__ == "__main__":
    test_returns_empty_with_fewer_than_3_upvotes()
    test_returns_clusters_with_enough_upvotes()
    test_cluster_count_bounded_by_3()
    test_each_interest_has_required_fields()
    test_no_crash_with_exact_3_upvotes()
    print("All interest clustering tests passed!")
