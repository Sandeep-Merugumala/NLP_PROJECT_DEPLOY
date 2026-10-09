# Scholar Inbox — New Feature Proposals

These are **new top-level features** (new tabs), not extensions of Daily Digest, Scholar Maps, Collections & Search, Conference Planner or AI Workspace.
Each one reuses assets that already exist in the project: TF-IDF / Gemini `text-embedding-004` embeddings, t-SNE coordinates, RAG chunks (Parent-Child), LangGraph agents, the Markdown-to-JSX compiler and the Gemini/Groq hybrid router.

Paths follow the layout used in `changes.md`: `backend/app.py`, `backend/core/`, `backend/agents/`, `frontend/src/components/`.

---

## Part 1 — Recommended Core Features (one per team member)

### 1. Research Trends Explorer

| Item | Detail |
|------|--------|
| **What the user sees** | A "Trends" tab with a stream/line chart of topic volume over time, a "Rising topics" list, and a "New this week" badge |
| **How it works** | Group papers by arXiv date and by cluster/topic label, compute growth rate (this week vs previous weeks), flag bursts. An LLM writes a one-paragraph "what's happening in X" summary |
| **Reuses** | t-SNE cluster labels, arXiv metadata, Gemini/Groq router |
| **Backend** | `backend/core/trends.py`, `GET /api/trends` in `app.py` |
| **Frontend** | `frontend/src/components/Trends.jsx` |
| **Output metrics** | Topic growth rate, burst detection, weekly volume per topic |

### 2. Research Gap and Idea Finder

| Item | Detail |
|------|--------|
| **What the user sees** | An "Ideas" tab: pick a topic area, see a highlighted low-density region on the 2D map, and get 3 LLM-generated research ideas with the supporting papers |
| **How it works** | Find sparse regions between clusters in t-SNE space, take the nearest papers on each side, and prompt the LLM to propose a combination or gap. Novelty score = 1 − max cosine similarity to the corpus |
| **Reuses** | Scholar Maps canvas, t-SNE coordinates, embeddings |
| **Backend** | `backend/core/gap_finder.py`, `POST /api/ideas` |
| **Frontend** | `frontend/src/components/IdeaFinder.jsx` (reuse the Maps canvas component) |
| **Output metrics** | Novelty score per idea, supporting-paper list |

### 3. Citation Finder / Related Work Writer

| Item | Detail |
|------|--------|
| **What the user sees** | A "Write" tab: paste a draft paragraph and get suggested papers per sentence, ready BibTeX, and an optional auto-drafted related-work paragraph |
| **How it works** | Split text into sentences, embed each sentence, retrieve top-k papers, have the LLM rewrite the paragraph with citations. The Scholar Inbox paper names missed-citation search as a core use case |
| **Reuses** | Semantic search, BibTeX export, embeddings, LLM router |
| **Backend** | `backend/core/citation_finder.py`, `POST /api/cite-suggest` |
| **Frontend** | `frontend/src/components/CiteFinder.jsx` |
| **Output metrics** | Recall@k on a small set of known citations |

### 4. Paper Comparison Studio

| Item | Detail |
|------|--------|
| **What the user sees** | A "Compare" tab: select 2–3 papers and get a side-by-side table (problem, method, dataset, results, limitations), a similarity score and "when to use which" |
| **How it works** | Pull abstract and PDF chunks from the RAG pipeline, extract fields with the LLM into JSON, render a table with the existing Markdown-to-JSX compiler |
| **Reuses** | RAG chunks, LangGraph, Markdown-to-JSX compiler |
| **Backend** | `backend/agents/compare_agent.py` (LangGraph node), `POST /api/compare` |
| **Frontend** | `frontend/src/components/Compare.jsx` |
| **Output metrics** | Field-extraction accuracy on a hand-labelled set of papers |

### 5. Study Mode (Quiz and Flashcards)

| Item | Detail |
|------|--------|
| **What the user sees** | A "Study" tab that generates MCQs and flashcards from any paper, tracks your score and schedules re-reviews |
| **How it works** | The LLM generates questions from the abstract and PDF chunks. Spaced repetition (SM-2) decides when each card returns |
| **Reuses** | RAG chunks, LLM router |
| **Backend** | `backend/core/study.py`, `GET/POST /api/study/*`, cards stored in `data/study.jsonl` |
| **Frontend** | `frontend/src/components/Study.jsx` |
| **Output metrics** | Quiz score over time, retention curve, question quality check |

---

## Part 2 — Extra Feature Ideas

| Feature | What it does | Where to implement |
|---------|--------------|--------------------|
| **Learning Path Generator** | For a topic, builds an ordered reading path from foundational to advanced using paper dates, similarity chains and LLM prerequisite judgement | `backend/core/learning_path.py`, `components/LearningPath.jsx` |
| **Notes Notebook (Second Brain)** | Notes attached to papers, semantically searchable and auto-linked to related papers | `data/notes.jsonl`, `components/Notebook.jsx` |
| **Paper-to-Content Studio** | One click turns a paper into an ELI5 summary, slide outline, tweet thread or podcast script, with Markdown export | `backend/agents/content_agent.py`, `components/ContentStudio.jsx` |
| **Audio Digest** | Reads the day's top papers aloud using the browser's `SpeechSynthesis`, with no backend cost | `DailyDigest.jsx` plus a small `AudioPlayer.jsx` |
| **Author and Venue Network** | Force-directed graph of co-authors and topic neighbours, optionally using the Semantic Scholar API for citations | `backend/core/network.py`, `components/Network.jsx` |
| **Watchlists and Alerts** | Follow keywords or authors, get a new-paper feed with a notification badge | `data/watchlists.json`, `components/Watchlist.jsx` |

---

## Part 3 — Where to Put Reinforcement Learning

`changes.md` removed the larger RL tasks as over-engineered for ~800 papers, but the teacher asked for RL somewhere. Keep at least one small, honest RL piece.

**Best fit: Study Mode (Feature 5).** Treat "which card to show next" as a bandit / RL problem:

- **State:** the user's per-card history (correct/incorrect, time since last review) and per-topic mastery estimate
- **Action:** which card to show next
- **Reward:** +1 for a correct answer at the learning frontier (not too easy, not too hard), 0 or negative for trivial or impossible cards
- **Method:** Thompson Sampling or LinUCB over cards, with SM-2 as the baseline
- **Evaluation:** compare RL vs SM-2 vs random ordering on simulated learners and on a short real session
- **Why it works at this scale:** it needs only dozens of interactions per user, which avoids the cold-start problem raised in `changes.md`

**Alternative placements** (if preferred):
- **Trends Explorer:** bandit that chooses which rising topics to surface in "Rising topics" (reward = clicks)
- **Idea Finder:** bandit that learns which kinds of gaps the user finds useful (reward = saved ideas)

---

## Part 4 — Suggested Split and Priority

| # | Feature | Complexity | Visible UI impact | RL option |
|---|---------|-----------|-------------------|-----------|
| 1 | Research Trends Explorer | Medium | High (charts) | Optional |
| 2 | Research Gap and Idea Finder | Medium–High | High (map overlay) | Optional |
| 3 | Citation Finder / Related Work Writer | Medium | High (practical) | No |
| 4 | Paper Comparison Studio | Medium | High (tables) | No |
| 5 | Study Mode | Medium | High (interactive) | **Yes (primary)** |

**Priority if time is short:** 5 → 3 → 4 → 1 → 2.

**Common to all features:**
- Add each new tab to the `App.jsx` navigation
- Log events with the reward logger (`data/reward_log.jsonl`) so results can be evaluated
- Add one test file per feature in `backend/tests/`
- Add a section per feature in `TECHNICAL.md`
- Keep the LLM calls behind the existing Gemini/Groq router for high availability
