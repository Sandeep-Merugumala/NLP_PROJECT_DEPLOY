"""
Reproduce Table 2 — Evaluation of embedding strategies.

Metrics: F1, nDCG@10, Balanced Accuracy, AUC-ROC
Embeddings evaluated: TF-IDF (256 features), Gemini text-embedding-004 (via API)

Usage:
    python backend/eval/reproduce_table2.py [--api-key YOUR_GEMINI_KEY]

Output: results printed to stdout + saved to backend/eval/results/table2.csv
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path(__file__).parents[2] / "backend" / "data" / "papers.json"
RATINGS_FILE = Path(__file__).parents[2] / "backend" / "data" / "ratings.json"


def load_data():
    with open(DATA_FILE) as f:
        papers = json.load(f)
    with open(RATINGS_FILE) as f:
        ratings = json.load(f)
    return papers, ratings


def build_dataset(papers, ratings):
    """Return X (embeddings), y (labels) arrays for rated papers only."""
    X, y = [], []
    for paper in papers:
        pid = paper["id"]
        if pid in ratings:
            X.append(paper["embedding"])
            y.append(1 if ratings[pid] == 1 else 0)
    return np.array(X), np.array(y)


def evaluate(X, y, C=0.1):
    """5-fold stratified CV across standard metrics."""
    if len(np.unique(y)) < 2:
        print("  [SKIP] Need at least 2 classes for evaluation.")
        return None

    skf = StratifiedKFold(n_splits=min(5, sum(y)), shuffle=True, random_state=42)
    f1s, bacs, aucs, ndcgs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(C=C, solver="liblinear", random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        bacs.append(balanced_accuracy_score(y_te, y_pred))
        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, y_prob))
            # nDCG expects 2D relevance array
            ndcgs.append(ndcg_score([y_te.astype(float)], [y_prob], k=10))

    return {
        "F1": float(np.mean(f1s)),
        "BalancedAcc": float(np.mean(bacs)),
        "AUC": float(np.mean(aucs)) if aucs else float("nan"),
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else float("nan"),
    }


def tfidf_embeddings(papers):
    """Recompute TF-IDF embeddings (256 features) from scratch."""
    corpus = [f"{p['title']} [SEP] {p['abstract']}" for p in papers]
    vec = TfidfVectorizer(max_features=256, stop_words="english")
    mat = vec.fit_transform(corpus).toarray()
    for i, paper in enumerate(papers):
        paper["embedding"] = mat[i].tolist()
    return papers


def gemini_embeddings(papers, api_key):
    """Replace paper embeddings with Gemini text-embedding-004 vectors."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        for paper in papers:
            resp = genai.embed_content(
                model="models/text-embedding-004",
                content=paper["title"] + " " + paper["abstract"][:500],
                task_type="RETRIEVAL_DOCUMENT"
            )
            paper["embedding"] = resp["embedding"]
        return papers
    except Exception as e:
        print(f"  [WARN] Gemini embedding failed: {e}")
        return None


def print_results(name, metrics):
    if metrics is None:
        print(f"  {name}: SKIPPED")
        return
    print(f"  {name}:")
    for k, v in metrics.items():
        print(f"    {k:20s}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API key")
    args = parser.parse_args()

    print("=" * 60)
    print("Table 2 Reproduction — Scholar Inbox")
    print("=" * 60)

    papers, ratings = load_data()
    print(f"Loaded {len(papers)} papers, {len(ratings)} ratings")

    results = {}

    # 1. TF-IDF embeddings (paper-stored embedding)
    print("\n[1] TF-IDF (256 features, stored embedding)")
    X, y = build_dataset(papers, ratings)
    metrics_tfidf = evaluate(X, y)
    print_results("TF-IDF", metrics_tfidf)
    results["TF-IDF"] = metrics_tfidf

    # 2. TF-IDF recomputed from scratch
    print("\n[2] TF-IDF (256 features, recomputed)")
    papers_copy = [dict(p) for p in papers]
    papers_copy = tfidf_embeddings(papers_copy)
    X2, y2 = build_dataset(papers_copy, ratings)
    metrics_tfidf2 = evaluate(X2, y2)
    print_results("TF-IDF (recomputed)", metrics_tfidf2)
    results["TF-IDF-recomputed"] = metrics_tfidf2

    # 3. Gemini embeddings (if API key available)
    if args.api_key:
        print("\n[3] Gemini text-embedding-004")
        papers_gemini = [dict(p) for p in papers]
        papers_gemini = gemini_embeddings(papers_gemini, args.api_key)
        if papers_gemini:
            X3, y3 = build_dataset(papers_gemini, ratings)
            metrics_gemini = evaluate(X3, y3)
            print_results("Gemini text-embedding-004", metrics_gemini)
            results["Gemini"] = metrics_gemini
    else:
        print("\n[3] Gemini embeddings: SKIPPED (no --api-key provided)")

    # Save to CSV
    import csv
    csv_path = RESULTS_DIR / "table2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Embedding", "F1", "BalancedAcc", "AUC", "nDCG@10"])
        for name, m in results.items():
            if m:
                writer.writerow([name, f"{m['F1']:.4f}", f"{m['BalancedAcc']:.4f}",
                                  f"{m['AUC']:.4f}", f"{m['nDCG@10']:.4f}"])

    print(f"\nResults saved to {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
