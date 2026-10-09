"""
Unit tests for backend/core/mmr.py — MMR re-ranking function.
Run: python -m pytest backend/tests/test_mmr.py -v
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.mmr import mmr_rerank


def make_papers(n):
    """Create n fake paper dicts with descending relevance scores."""
    return [
        {"id": f"paper_{i}", "title": f"Paper {i}", "relevance_score": float(n - 1 - i)}
        for i in range(n)
    ]


def make_identity_matrix(n):
    """Dense identity matrix — each paper is maximally dissimilar to all others."""
    return np.eye(n)


def make_constant_matrix(n):
    """All papers identical — similarity is 1 everywhere."""
    return np.ones((n, n))


def test_lambda1_returns_relevance_order():
    """λ=1 should return papers sorted purely by relevance (no diversity applied)."""
    papers = make_papers(10)
    matrix = make_identity_matrix(10)
    index_map = {f"paper_{i}": i for i in range(10)}

    result = mmr_rerank(papers, matrix, index_map, lam=1.0, top_k=5)
    scores = [p["relevance_score"] for p in result]
    assert scores == sorted(scores, reverse=True), "λ=1 should preserve relevance order"


def test_top_k_limits_output():
    """Output should never exceed top_k."""
    papers = make_papers(20)
    matrix = make_identity_matrix(20)
    index_map = {f"paper_{i}": i for i in range(20)}

    for k in [3, 7, 15]:
        result = mmr_rerank(papers, matrix, index_map, lam=0.5, top_k=k)
        assert len(result) <= k, f"top_k={k} violated: got {len(result)}"


def test_diversity_selects_different_papers():
    """
    With identical-vector papers and λ=0 (max diversity), subsequent picks should
    still produce a valid list of unique papers.
    """
    papers = make_papers(8)
    matrix = make_constant_matrix(8)  # all papers look the same
    index_map = {f"paper_{i}": i for i in range(8)}

    result = mmr_rerank(papers, matrix, index_map, lam=0.01, top_k=5)
    ids = [p["id"] for p in result]
    assert len(ids) == len(set(ids)), "Duplicate papers in MMR result"


def test_no_candidates_returns_empty():
    """Empty candidate list should return empty list."""
    result = mmr_rerank([], np.eye(5), {}, lam=0.7, top_k=5)
    assert result == []


def test_single_candidate():
    """Single candidate should be returned as-is."""
    papers = make_papers(1)
    matrix = make_identity_matrix(1)
    index_map = {"paper_0": 0}

    result = mmr_rerank(papers, matrix, index_map, lam=0.7, top_k=5)
    assert len(result) == 1
    assert result[0]["id"] == "paper_0"


def test_missing_id_skipped():
    """Papers whose IDs are absent from paper_index_map should be silently skipped."""
    papers = make_papers(5)
    matrix = make_identity_matrix(5)
    # Only map 3 of the 5 papers
    index_map = {f"paper_{i}": i for i in range(3)}

    result = mmr_rerank(papers, matrix, index_map, lam=0.7, top_k=5)
    for p in result:
        assert p["id"] in index_map, f"Unexpected paper: {p['id']}"


if __name__ == "__main__":
    test_lambda1_returns_relevance_order()
    test_top_k_limits_output()
    test_diversity_selects_different_papers()
    test_no_candidates_returns_empty()
    test_single_candidate()
    test_missing_id_skipped()
    print("All MMR tests passed!")
