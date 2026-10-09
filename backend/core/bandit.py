"""
Contextual bandit (LinUCB) exploration layer — the reinforcement-learning
component of the recommender.

The logistic-regression recommender is pure *exploitation*: it always ranks by
predicted relevance, so you only ever see papers near what you already liked.
LinUCB adds *exploration*: it keeps an online linear model of reward and, for
each candidate, adds an "uncertainty bonus" so papers the model is unsure about
(but might love) get surfaced too. This is the exploration/exploitation trade-off
at the heart of reinforcement learning, and it fits this project's data scale
(it learns from a few dozen interactions, no cold-start blow-up).

State : per-paper feature vector x (semantic or TF-IDF embedding)
Action: which paper to surface
Reward: +1 for a liked paper, 0 for a disliked one
Model : A = I + Σ x xᵀ ,  b = Σ r x ,  θ = A⁻¹ b
Score : θ·x  +  α · sqrt(xᵀ A⁻¹ x)     (estimate + confidence bound)
"""

import numpy as np


class LinUCB:
    def __init__(self, dim, alpha=0.6, l2=1.0):
        self.dim = dim
        self.alpha = alpha
        self.A = np.eye(dim) * l2
        self.b = np.zeros(dim)

    def update(self, x, reward):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            return
        self.A += np.outer(x, x)
        self.b += reward * x

    def score_many(self, X):
        """Return UCB scores and the exploration-bonus part for a matrix X (n×dim)."""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        means = X @ theta                              # exploitation
        # uncertainty bonus: sqrt(x A^-1 x) per row
        bonus = np.sqrt(np.einsum("ij,jk,ik->i", X, A_inv, X).clip(min=0))
        return means + self.alpha * bonus, means, self.alpha * bonus


def _feat(paper, feature_key, vectors=None, vdim=None):
    if vectors is not None and vdim:
        v = vectors.get(paper["id"])
        return np.asarray(v if v is not None else [0.0] * vdim, dtype=float)
    v = paper.get(feature_key)
    if v is None:
        v = paper.get("embedding")
    return np.asarray(v, dtype=float)


def rank_with_bandit(candidates, all_papers, ratings, feature_key="embedding", alpha=0.6, vectors=None):
    """
    Re-rank `candidates` with a LinUCB bandit trained on the user's ratings.
    Returns the candidates ordered by UCB score, each annotated with
    `ucb_score` and `explore_bonus` (how much exploration contributed).
    Falls back to the given order if features are unusable.
    """
    if not candidates:
        return candidates
    vdim = None
    if vectors:
        try:
            vdim = len(next(iter(vectors.values())))
        except StopIteration:
            vdim = None
    try:
        dim = len(_feat(candidates[0], feature_key, vectors, vdim))
    except Exception:
        return candidates
    if dim == 0:
        return candidates

    model = LinUCB(dim, alpha=alpha)
    # Train on every rated paper we can find features for.
    by_id = {p["id"]: p for p in all_papers}
    for pid, r in ratings.items():
        p = by_id.get(pid)
        if p is None:
            continue
        reward = 1.0 if r == 1 else 0.0
        try:
            model.update(_feat(p, feature_key, vectors, vdim), reward)
        except Exception:
            continue

    try:
        X = np.vstack([_feat(c, feature_key, vectors, vdim) for c in candidates])
    except Exception:
        return candidates
    if X.shape[1] != dim:
        return candidates

    ucb, means, bonus = model.score_many(X)
    order = np.argsort(-ucb)
    ranked = []
    for idx in order:
        c = dict(candidates[idx])
        c["ucb_score"] = float(round(ucb[idx], 4))
        c["explore_bonus"] = float(round(bonus[idx], 4))
        ranked.append(c)
    return ranked
