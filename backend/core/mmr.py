"""
Maximal Marginal Relevance (MMR) re-ranking.

MMR balances relevance and diversity in ranked lists:
    MMR_score(d) = λ * relevance(d) - (1 - λ) * max_sim(d, selected)

Parameters
----------
λ (lam) : float in [0, 1]
    λ = 1  → pure relevance ranking (identical to original order)
    λ = 0  → maximum diversity (greedy farthest-first)
"""

import numpy as np


def mmr_rerank(candidates, tfidf_matrix, paper_index_map, lam=0.7, top_k=15):
    """
    Re-rank a list of candidate papers using Maximal Marginal Relevance.

    Parameters
    ----------
    candidates : list of paper dicts
        Already sorted by relevance_score descending.
    tfidf_matrix : np.ndarray (n_papers x n_features), dense
        Global TF-IDF matrix for ALL papers in the corpus.
    paper_index_map : dict {paper_id: row_index}
        Maps paper IDs to their row in tfidf_matrix.
    lam : float
        Trade-off parameter λ ∈ [0, 1].
    top_k : int
        Number of papers to return.

    Returns
    -------
    list of paper dicts in MMR order.
    """
    if not candidates or lam >= 1.0:
        # λ=1 → pure relevance, no MMR computation needed
        return candidates[:top_k]

    if lam <= 0.0:
        # λ=0 → pure diversity via greedy farthest-first from corpus centroid
        lam = 0.001  # avoid division edge-cases; effective diversity mode

    # Gather indices and embeddings for candidate papers
    indices = []
    valid_candidates = []
    for p in candidates:
        idx = paper_index_map.get(p["id"])
        if idx is not None:
            indices.append(idx)
            valid_candidates.append(p)

    if not valid_candidates:
        return candidates[:top_k]

    # Candidate embedding matrix (rows = candidates)
    cand_vecs = tfidf_matrix[indices]  # shape: (n_cands, n_features)

    # Normalise rows for cosine similarity
    norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cand_vecs_norm = cand_vecs / norms

    # Normalise relevance scores to [0, 1]
    scores = np.array([p["relevance_score"] for p in valid_candidates])
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        rel_scores = (scores - score_min) / (score_max - score_min)
    else:
        rel_scores = np.ones(len(scores)) * 0.5

    selected_indices = []   # indices into valid_candidates
    selected_vecs = []       # normalised embeddings of selected papers

    remaining = list(range(len(valid_candidates)))

    for _ in range(min(top_k, len(valid_candidates))):
        best_idx = None
        best_score = -np.inf

        for r in remaining:
            rel = rel_scores[r]

            # Similarity to already-selected papers
            if selected_vecs:
                sim_to_selected = float(np.max(
                    cand_vecs_norm[r] @ np.array(selected_vecs).T
                ))
            else:
                sim_to_selected = 0.0

            mmr_s = lam * rel - (1.0 - lam) * sim_to_selected

            if mmr_s > best_score:
                best_score = mmr_s
                best_idx = r

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        selected_vecs.append(cand_vecs_norm[best_idx])
        remaining.remove(best_idx)

    return [valid_candidates[i] for i in selected_indices]
