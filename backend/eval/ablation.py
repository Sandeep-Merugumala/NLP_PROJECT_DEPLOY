"""
Ablation Study — Sweep C, V, S hyperparameters.

Sweeps:
  C ∈ {0.01, 0.1, 1.0}        (logistic regression regularisation)
  V ∈ {0.5, 0.8, 1.0}         (explicit-negative weight ratio)
  S ∈ {1.0, 5.0, 10.0}        (negative class scale factor)

Also sweeps PCA dimensionality applied to TF-IDF features: {16, 32, 64, 128, 256}.

Usage:
    python backend/eval/ablation.py

Output: backend/eval/results/ablation_hparams.csv  &  ablation_pca.csv
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

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
    X, y = [], []
    for paper in papers:
        pid = paper["id"]
        if pid in ratings:
            X.append(paper["embedding"])
            y.append(1 if ratings[pid] == 1 else 0)
    return np.array(X), np.array(y)


def weighted_train(X_tr, y_tr, C, V, S):
    """Train LogReg with the paper's custom weighting scheme."""
    import random as rnd
    positive = [(x, 1) for x, label in zip(X_tr, y_tr) if label == 1]
    negative = [(x, 0) for x, label in zip(X_tr, y_tr) if label == 0]

    n_P = len(positive)
    n_N = len(negative)
    n_R = min(len(negative), 100)
    random_neg = rnd.sample(negative, n_R) if n_R > 0 else []

    all_X, all_y, all_w = [], [], []
    n_T = n_P + n_N + n_R
    denom = V * n_N + (1.0 - V) * n_R or 1.0
    w_P = n_T / max(n_P, 1)
    w_N = n_T * (S * V) / denom
    w_R = n_T * (S * (1.0 - V)) / denom

    for x, _ in positive:
        all_X.append(x); all_y.append(1); all_w.append(w_P)
    for x, _ in negative:
        all_X.append(x); all_y.append(0); all_w.append(w_N)
    for x, _ in random_neg:
        all_X.append(x); all_y.append(0); all_w.append(w_R)

    if len(np.unique(all_y)) < 2:
        return None

    clf = LogisticRegression(C=C, solver="liblinear", random_state=42)
    clf.fit(np.array(all_X), np.array(all_y), sample_weight=np.array(all_w))
    return clf


def cross_val_metrics(X, y, C=0.1, V=0.8, S=5.0):
    if len(np.unique(y)) < 2 or len(y) < 4:
        return None
    skf = StratifiedKFold(n_splits=min(5, sum(y)), shuffle=True, random_state=42)
    bacs, f1s = [], []
    for tr, te in skf.split(X, y):
        clf = weighted_train(X[tr], y[tr], C, V, S)
        if clf is None:
            continue
        y_pred = clf.predict(X[te])
        bacs.append(balanced_accuracy_score(y[te], y_pred))
        f1s.append(f1_score(y[te], y_pred, zero_division=0))
    if not bacs:
        return None
    return {"BalancedAcc": float(np.mean(bacs)), "F1": float(np.mean(f1s))}


def run_hparam_sweep(papers, ratings):
    print("\n[Hyperparameter Sweep] C × V × S")
    X, y = build_dataset(papers, ratings)
    rows = []
    for C, V, S in product([0.01, 0.1, 1.0], [0.5, 0.8, 1.0], [1.0, 5.0, 10.0]):
        m = cross_val_metrics(X, y, C, V, S)
        status = f"BAcc={m['BalancedAcc']:.4f} F1={m['F1']:.4f}" if m else "SKIPPED"
        print(f"  C={C:.2f} V={V:.1f} S={S:.1f}  → {status}")
        rows.append({"C": C, "V": V, "S": S,
                     "BalancedAcc": m["BalancedAcc"] if m else float("nan"),
                     "F1": m["F1"] if m else float("nan")})

    import csv
    path = RESULTS_DIR / "ablation_hparams.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["C", "V", "S", "BalancedAcc", "F1"])
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


def run_pca_sweep(papers, ratings):
    print("\n[PCA Sweep] n_components ∈ {16, 32, 64, 128, 256}")
    X_full, y = build_dataset(papers, ratings)
    rows = []
    for n_comp in [16, 32, 64, 128, 256]:
        if n_comp >= X_full.shape[1]:
            X_red = X_full
        else:
            pca = PCA(n_components=n_comp, random_state=42)
            X_red = pca.fit_transform(X_full)
        m = cross_val_metrics(X_red, y)
        status = f"BAcc={m['BalancedAcc']:.4f} F1={m['F1']:.4f}" if m else "SKIPPED"
        print(f"  PCA n={n_comp:3d}  → {status}")
        rows.append({"n_components": n_comp,
                     "BalancedAcc": m["BalancedAcc"] if m else float("nan"),
                     "F1": m["F1"] if m else float("nan")})

    import csv
    path = RESULTS_DIR / "ablation_pca.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_components", "BalancedAcc", "F1"])
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation Study — Scholar Inbox")
    print("=" * 60)
    papers, ratings = load_data()
    print(f"Loaded {len(papers)} papers, {len(ratings)} ratings")
    run_hparam_sweep(papers, ratings)
    run_pca_sweep(papers, ratings)
    print("\nDone.")
