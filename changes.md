# Scholar Inbox — Change Evaluation & Implementation Plan

## Overview

This document critically evaluates each of the 20 originally planned tasks against the **real codebase**
(Flask + React, ~800 papers, no existing RL infra, simple LogisticRegression recommender)
and then defines a pragmatic, phased implementation roadmap that maximises **visible UI impact**.

---

## Section 1 — Original Tasks: Verdict Table

| # | Task | Verdict | Reason |
|---|------|---------|--------|
| 1 | Contextual Bandit Digest Ranker (LinUCB/Thompson) | ⚠️ **Simplify** | Full bandit training requires thousands of interactions. With 800 papers and cold-start users, LinUCB will barely beat the existing LogReg. Replace with **MMR + relevance blend** (Task 8) which gives identical diversity benefits with zero cold-start problem. |
| 2 | RL Active Learning Policy | ❌ **Remove** | Existing K-Means boundary sampling already does what an RL policy would do at this scale. An RL policy needs episodic reward signals that do not exist yet. Over-engineered. Keep K-Means. |
| 3 | User Simulator / Gym Env | ❌ **Remove** | Requires a labelled ground-truth dataset (800k). We only have 800 papers and no user logs. The simulator would be synthetic noise. Not worth the complexity. |
| 4 | Off-Policy Evaluation (IPS/DR) | ❌ **Remove** | OPE is only meaningful when you have a logged behaviour policy. We have none. Remove entirely. |
| 5 | Conference Planner RL Scheduler | ❌ **Remove** | The greedy keyword + relevance ranking in `app.py` already works well and the UI is functional. Q-learning on 4 static sessions adds no value. |
| 6 | Query-Reformulation Bandit | ⚠️ **Simplify** | Replace the full bandit with a lightweight **query expansion** step: extract top TF-IDF terms from upvoted papers and append them to the search. Visible improvement, zero bandit complexity. |
| 7 | Per-User Hyperparameter Bandit | ❌ **Remove** | Hyperparameter tuning is a backend concern invisible to the user. The existing C/V/S defaults already work. Remove. |
| 8 | MMR Diversity Re-ranking | ✅ **Keep** | High value. Simple to implement (cosine + λ trade-off). Immediately improves Digest quality and is **visible to the user** as a "Diversity" slider. |
| 9 | Multi-Interest Profiles | ✅ **Keep** | Cluster upvoted paper embeddings into 2–3 interest topics. Show topic chips in the Digest sidebar. Very high UI impact. |
| 10 | Hybrid Two-Tower Model | ❌ **Remove** | Requires a large interaction matrix (thousands of ratings). We have at most dozens. The existing LogReg on TF-IDF embeddings is the correct choice for this scale. |
| 11 | "Why Recommended" Explanations | ✅ **Keep** | Already partially present (`/api/explain`). Need to surface top contributing TF-IDF terms + nearest liked paper in the UI card. High demo value. |
| 12 | ANN Index (FAISS) for Similar Papers | ⚠️ **Simplify** | FAISS is overkill for 800 papers. Use **numpy cosine similarity** (already done in search). Add a `/api/similar/<id>` endpoint and a "Similar Papers" panel in the UI. No FAISS needed. |
| 13 | Reproduce Table 2 (Eval Script) | ✅ **Keep** | Useful for the teacher demo. `backend/eval/reproduce_table2.py` — runs offline, no UI required. |
| 14 | Reproduce Ablations (PCA & C/V/S) | ✅ **Keep** | `backend/eval/ablation.py` — sweeps hyperparameters, saves CSV. Useful for the report. |
| 15 | RAG Chat Evaluation | ✅ **Keep** | `backend/eval/rag_eval.py` — automated precision@k and faithfulness test. Extends existing test suite. |
| 16 | Email Digest | ❌ **Remove** | Requires SMTP server, user accounts, and scheduler infrastructure. Out of scope for a demo. Replace with an **in-app notification badge** (new Feature A). |
| 17 | Feedback & Reward Logging | ✅ **Keep** | Low-effort, high-value. Log every rate/bookmark/click event to `data/reward_log.jsonl`. Powers future RL and evaluation. |
| 18 | Testing | ✅ **Keep** | Extend `backend/tests/` to cover new modules added in this plan. |
| 19 | Documentation Updates | ✅ **Keep** | Update `TECHNICAL.md` with new architecture diagram, formula updates, and results table. |
| 20 | Docker Compose Updates | ✅ **Keep** | Add named volumes for `data/` and `pdf_cache/`. No scheduler service needed (email removed). |

---

## Section 2 — New Features (UI-Visible, High Impact)

These replace the removed tasks and fill genuine gaps in the interface.

### Feature A: Live Stats Dashboard (Sidebar Widget)
- **What:** A compact stats panel in the App sidebar showing: total papers loaded, # rated, # bookmarked, top category, and a mini sparkline of daily rating activity.
- **Why:** Makes the system feel alive and personalised immediately on open.
- **UI Location:** App.jsx sidebar / top bar.
- **Backend:** Extend `/api/papers` response with aggregated stats or add `GET /api/stats`.

### Feature B: Similar Papers Panel ("More Like This")
- **What:** Clicking a paper shows a side-drawer with 5 similar papers ranked by cosine similarity on TF-IDF vectors (already computed in memory). No FAISS needed.
- **Why:** Directly reuses `tfidf_matrix` already in `app.py`. Zero new ML. High user value.
- **UI Location:** Paper card → "More Like This" button → side drawer.
- **Backend:** New endpoint `GET /api/similar/<paper_id>`.

### Feature C: MMR Diversity Slider in Digest
- **What:** A slider (λ 0→1) in the Digest header. λ=1 = pure relevance, λ=0 = maximum diversity. Backend applies MMR re-ranking.
- **Why:** Directly replaces Tasks 1 & 8. Visible, interactive, and academically sound.
- **UI Location:** `DailyDigest.jsx` header controls.
- **Backend:** `GET /api/digest?range=day&diversity=0.5` → `backend/core/mmr.py`.

### Feature D: Interest Profile Chips (Multi-Interest)
- **What:** After ≥3 upvotes, K-Means (k=2 or 3) clusters the upvoted paper embeddings. The Digest sidebar shows coloured topic chips (e.g. "Topic 1: Transformers", "Topic 2: Computer Vision"). Clicking a chip filters the digest.
- **Why:** Replaces Task 9. Visible personalisation that shows the recommender is learning.
- **UI Location:** `DailyDigest.jsx` sidebar.
- **Backend:** Extend `GET /api/digest` response with `interests` field, or new `GET /api/interests`.

### Feature E: Explanation Popover ("Why Recommended")
- **What:** Each Digest paper card has a small ℹ️ icon. Clicking it shows a popover with: top 5 contributing TF-IDF keywords, nearest liked paper title, and relevance score bar.
- **Why:** Task 11 already has `GET /api/explain` but it's not wired to any UI element. This closes the gap.
- **UI Location:** `DailyDigest.jsx` paper card.
- **Backend:** Already exists (`/api/explain`). Just wire the frontend.

### Feature F: Reading History & Bookmark Timeline
- **What:** A new section inside Collections showing a chronological timeline of rated/bookmarked papers with timestamps and rating icons.
- **Why:** Reward logging (Task 17) captures timestamps. Displaying them creates a personalised history view with zero extra backend effort.
- **UI Location:** New section inside `Collections.jsx`.
- **Backend:** `GET /api/history` reads from `data/reward_log.jsonl`.

### Feature G: Search Autocomplete with Query Expansion
- **What:** The search bar auto-suggests completions using top TF-IDF terms from the corpus. If user has upvotes, the query is silently expanded with their top interest terms (replaces Task 6 bandit).
- **Why:** Replaces the complex query-reformulation bandit with a practical, visible feature.
- **UI Location:** `Collections.jsx` search bar.
- **Backend:** Extend `/api/search` to accept `expand=true` and extract top terms from upvoted paper embeddings.

---

## Section 3 — Implementation Plan (Phased)

### Phase 1 — Backend Foundation (Day 1–2)

| Step | File | Description |
|------|------|-------------|
| 1.1 | `backend/core/mmr.py` | Implement MMR re-ranking: `mmr(candidates, tfidf_matrix, λ)` returns diversified list |
| 1.2 | `backend/app.py` | Update `/api/digest` to accept `?diversity=λ` and call `mmr.py` |
| 1.3 | `backend/app.py` | Add `GET /api/similar/<paper_id>` using numpy cosine on `tfidf_matrix` |
| 1.4 | `backend/app.py` | Add `GET /api/interests` — K-Means on upvoted embeddings, return cluster labels + top terms |
| 1.5 | `backend/app.py` | Add `GET /api/stats` — aggregate counts: total, rated, bookmarked, top category |
| 1.6 | `backend/app.py` | Add `POST /api/log-event` + `GET /api/history` wired to `data/reward_log.jsonl` |
| 1.7 | `backend/app.py` | Extend `/api/search` with `expand=true` query expansion from upvoted paper terms |

### Phase 2 — Frontend UI (Day 3–4)

| Step | File | Description |
|------|------|-------------|
| 2.1 | `frontend/src/App.jsx` | Add Stats widget to sidebar (calls `/api/stats`) |
| 2.2 | `frontend/src/components/DailyDigest.jsx` | Add MMR diversity slider to header |
| 2.3 | `frontend/src/components/DailyDigest.jsx` | Add Interest Profile chips in sidebar (calls `/api/interests`) |
| 2.4 | `frontend/src/components/DailyDigest.jsx` | Wire ℹ️ popover on each card to `/api/explain` (Feature E) |
| 2.5 | `frontend/src/components/DailyDigest.jsx` | Add "More Like This" button → side drawer (calls `/api/similar/<id>`) |
| 2.6 | `frontend/src/components/Collections.jsx` | Add search autocomplete with TF-IDF suggestions + query expansion |
| 2.7 | `frontend/src/components/Collections.jsx` | Add Reading History timeline section |

### Phase 3 — Evaluation Scripts (Day 5)

| Step | File | Description |
|------|------|-------------|
| 3.1 | `backend/eval/reproduce_table2.py` | Evaluate TF-IDF vs Gemini embeddings on F1, nDCG, Balanced Accuracy, AUC |
| 3.2 | `backend/eval/ablation.py` | Sweep C ∈ {0.01,0.1,1}, V ∈ {0.5,0.8,1.0}, S ∈ {1,5,10}; save results CSV |
| 3.3 | `backend/eval/rag_eval.py` | Test RAG precision@k, faithfulness score (LLM-as-judge or ROUGE) |

### Phase 4 — Tests, Docs & Docker (Day 6)

| Step | File | Description |
|------|------|-------------|
| 4.1 | `backend/tests/test_mmr.py` | Unit tests for MMR function (diversity guarantee, boundary λ values) |
| 4.2 | `backend/tests/test_interests.py` | Unit tests for interest clustering (min upvotes, cluster count) |
| 4.3 | `backend/tests/test_history.py` | Unit tests for event logging and history retrieval |
| 4.4 | `TECHNICAL.md` | Update architecture diagram, add MMR formula, new endpoints table |
| 4.5 | `docker-compose.yml` | Add named volumes for `data/` and `pdf_cache/` |

---

## Section 4 — Summary of Decisions

### ❌ Removed (7 tasks): Tasks 2, 3, 4, 5, 7, 10, 16
These were either:
- **Over-engineered** for the dataset scale (RL policies, Two-Tower, OPE, Gym env)
- **Infrastructure-heavy** with no user-facing payoff (Email Digest)
- **Redundant** with existing working logic (Conference RL, HParam bandit)

### ⚠️ Simplified (3 tasks):
- Task 1 → Feature C (MMR Diversity Slider — same academic value, no cold-start)
- Task 6 → Feature G (Query Expansion — same goal, simpler implementation)
- Task 12 → Feature B (numpy cosine Similar Papers — no FAISS needed at 800 papers)

### ✅ Kept (11 tasks): 8, 9, 11, 13, 14, 15, 17, 18, 19, 20 (implemented as Phases 1–4)

### 🆕 New Features (7): A (Stats Dashboard), B (Similar Papers), C (MMR Slider), D (Interest Chips), E (Explanation Popover), F (History Timeline), G (Search Autocomplete)

---

## Dependency Map

```
Phase 1 (Backend)
  ├── 1.1 mmr.py ──────────────────► 2.2 MMR Slider (Frontend)
  ├── 1.3 /api/similar ────────────► 2.5 More Like This (Frontend)
  ├── 1.4 /api/interests ──────────► 2.3 Interest Chips (Frontend)
  ├── 1.5 /api/stats ──────────────► 2.1 Stats Widget (Frontend)
  ├── 1.6 /api/history ────────────► 2.7 History Timeline (Frontend)
  └── 1.7 Search Expansion ────────► 2.6 Autocomplete (Frontend)

Phase 3 (Eval) — standalone, no UI deps
Phase 4 (Tests/Docs) — depends on Phase 1 + 3
```
