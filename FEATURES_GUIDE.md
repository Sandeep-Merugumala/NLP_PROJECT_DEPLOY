# Scholar Inbox - New Features Implementation Guide

This document describes the 5 new core features added to Scholar Inbox, their architecture, and usage patterns.

---

## Table of Contents

1. [Research Trends Explorer](#1-research-trends-explorer)
2. [Research Gap and Idea Finder](#2-research-gap-and-idea-finder)
3. [Citation Finder & Related Work Writer](#3-citation-finder--related-work-writer)
4. [Paper Comparison Studio](#4-paper-comparison-studio)
5. [Study Mode](#5-study-mode)
6. [Cross-Cutting Requirements](#cross-cutting-requirements)
7. [API Reference](#api-reference)
8. [Frontend Integration](#frontend-integration)

---

## 1. Research Trends Explorer

### What It Does
Analyzes paper publication volume over time, detects rising topics, and generates LLM-powered summaries of trend insights.

### Architecture

**Backend: `backend/core/trends.py`**
- `compute_trends(papers_db, cluster_labels)` - Main function
  - Groups papers by week
  - Computes topic volume trends
  - Detects rising topics (>20% growth week-over-week)
  - Identifies "new this week" topics
  - Returns structured trend data

- `extract_topic_from_title(title)` - Topic classification
  - Keyword-based extraction
  - Fallback: "General Machine Learning"

- `generate_trend_summary(trends_data, api_key, use_groq)` - LLM summary
  - Uses Gemini/Groq hybrid router
  - Generates 2-3 sentence trend summary

**Frontend: `frontend/src/components/Trends.jsx`**
- Displays trend data in card layout
- Rising topics with growth rates
- Top topics overall
- New this week badges
- Refresh button to reload trends

### API Endpoint
```
GET /api/trends
Response: {
  "trend_data": [{"week": "2024-W01", "topics": {...}, "total_papers": 18}, ...],
  "rising_topics": [{"topic": "LLM", "growth_rate": 0.45, "this_week": 12, "prev_week": 8}, ...],
  "new_this_week": ["Topic1", "Topic2"],
  "top_topics": ["Machine Learning", "NLP", "Vision"],
  "trend_summary": "..."
}
```

### Usage Example
```python
from core.trends import compute_trends, generate_trend_summary

trends = compute_trends(papers_db)
summary = generate_trend_summary(trends, api_key)
```

---

## 2. Research Gap and Idea Finder

### What It Does
Identifies sparse (low-density) regions in the research embedding space and generates novel research ideas with supporting papers.

### Architecture

**Backend: `backend/core/gap_finder.py`**
- `find_sparse_regions(embeddings, t_sne_coords, num_regions, sparsity_threshold)`
  - Computes local density in embedding space
  - Identifies sparse points (high density threshold)
  - Clusters sparse points into regions
  - Returns region centers and neighboring papers

- `compute_novelty_score(paper_embedding, all_embeddings, top_k)`
  - Novelty = 1 - max_cosine_similarity
  - Lower = more novel/unique
  - Clamps to [0, 1]

- `generate_idea_suggestions(sparse_region, papers_db, embeddings, api_key)`
  - Analyzes papers at region boundary
  - Prompts LLM to generate 3 novel ideas
  - Computes novelty scores for each idea
  - Returns ideas with supporting papers

- `find_gap_for_topic(papers_db, embeddings, t_sne_coords, topic_keyword, api_key)`
  - Main entry point
  - Filters papers by keyword
  - Finds sparse regions
  - Generates ideas for top region

**Frontend: `frontend/src/components/IdeaFinder.jsx`**
- Topic keyword search input
- Interactive idea cards with novelty scores
- Supporting papers display
- Sparse region visualization

### API Endpoint
```
POST /api/ideas
Request: {"topic_keyword": "Large Language Models", "api_key": "..."}
Response: {
  "topic": "Large Language Models",
  "sparse_regions": [...],
  "ideas": [
    {"title": "...", "description": "..."},
    ...
  ],
  "novelty_scores": [0.75, 0.82, 0.68],
  "supporting_papers": [...]
}
```

---

## 3. Citation Finder & Related Work Writer

### What It Does
Finds relevant papers for text sections via semantic search, suggests citations, exports BibTeX, and auto-rewrites paragraphs with integrated citations.

### Architecture

**Backend: `backend/core/citation_finder.py`**
- `split_into_sentences(text)` - Sentence tokenization

- `find_related_papers(sentence, papers_db, tfidf_vectorizer, tfidf_matrix, top_k)`
  - Uses TF-IDF similarity
  - Returns top-k related papers with scores

- `generate_bibtex(paper)` - BibTeX generation
  - Standard format (title, authors, year, arXiv ID)
  - Handles special characters

- `suggest_citations_for_paragraph(paragraph_text, papers_db, tfidf_vectorizer, tfidf_matrix, top_k)`
  - Analyzes each sentence
  - Finds related papers per sentence
  - Deduplicates papers
  - Returns structured suggestions with BibTeX

- `rewrite_paragraph_with_citations(paragraph_text, citation_suggestions, api_key)`
  - Uses LLM to naturally integrate citations
  - Preserves original meaning/tone
  - Returns rewritten paragraph with [arxiv_id] citations

- `process_citation_request(user_text, papers_db, tfidf_vectorizer, tfidf_matrix, api_key)`
  - Main entry point
  - Processes paragraph end-to-end

**Frontend: `frontend/src/components/CiteFinder.jsx`**
- Textarea for paragraph input
- Three views: Suggestions, Rewritten, BibTeX
- Copy-to-clipboard for rewritten text
- Download .bib file button

### API Endpoint
```
POST /api/cite-suggest
Request: {"text": "Your paragraph here...", "api_key": "..."}
Response: {
  "original_text": "...",
  "citations": [
    {
      "text": "sentence",
      "suggested_papers": [{"id": "...", "title": "...", "similarity_score": 0.85}]
    },
    ...
  ],
  "rewritten_paragraph": "...",
  "bibtex": "@article{...}\n...",
  "statistics": {"papers_found": 5, "citations_suggested": 3, "sentences_analyzed": 4}
}
```

---

## 4. Paper Comparison Studio

### What It Does
Extracts structured information from 2-3 papers and renders a side-by-side comparison table with LLM-generated insights.

### Architecture

**Backend: `backend/agents/compare_agent.py` (LangGraph Node)**
- `ComparisonAgent` class
  - `extract_paper_info(paper)` - Uses LLM to extract:
    - Problem/Research Question
    - Methodology
    - Dataset/Benchmark
    - Key Results
    - Limitations
    - Venue
  
  - `compare_papers(paper_list)` - Main comparison logic
    - Extracts info from each paper (max 3)
    - Builds structured comparison table
    - Generates "when to use" recommendations
    - Computes similarity score (keyword overlap)

  - `_generate_when_to_use(papers_info)` - LLM recommendations
  - `_calculate_similarity(papers_info)` - Similarity heuristic

**Frontend: `frontend/src/components/Compare.jsx`**
- Paper selection (up to 3)
- Remove paper buttons
- Comparison table with all fields
- Similarity score display
- "When to use" recommendations

### API Endpoint
```
POST /api/compare
Request: {"paper_ids": ["2024.01.001", "2024.01.002"], "api_key": "..."}
Response: {
  "comparison_table": {
    "Paper": ["Title1...", "Title2..."],
    "Problem": ["Problem1", "Problem2"],
    "Methodology": ["Method1", "Method2"],
    ...
  },
  "papers": [{"title": "...", "year": "2024", ...}],
  "when_to_use": "Use Paper 1 for ... | Use Paper 2 for ...",
  "similarity_score": 0.65,
  "count": 2
}
```

---

## 5. Study Mode

### What It Does
Generates MCQs and flashcards from papers with SM-2 spaced repetition scheduling and Thompson Sampling RL for adaptive card selection.

### Architecture

**Backend: `backend/core/study.py`**

**Card Class**
- Flashcard/MCQ representation
- SM-2 parameters: `ease_factor`, `interval`, `repetitions`, `last_reviewed`, `next_review`
- RL parameters: `alpha`, `beta` (Thompson Sampling), `correct_count`, `incorrect_count`

**SM2Scheduler** (Spaced Repetition)
- `update_card(card, quality, days_since_last)`
  - Quality: 0-5 scale (0-1 = fail, 2-4 = hard/ok/easy, 5 = perfect)
  - Adjusts interval and ease_factor based on quality
  - Resets on failure, exponential growth on success
  - Follows original SM-2 algorithm

- `get_due_cards(cards)`
  - Returns cards due for review today

**RLCardScheduler** (Thompson Sampling Bandit)
- `sample_arm(card)`
  - Samples success probability from Beta(alpha, beta)

- `select_next_card(due_cards, topic_mastery)`
  - Uses Thompson Sampling to select next card
  - Prioritizes "learning frontier" (cards at ~50% success rate)
  - Combines TS with frontier bonus

- `update_card_rl(card, correct, reward)`
  - Updates alpha/beta based on correctness
  - Adjusts for learning frontier

**Card Generation**
- `generate_cards_from_paper(paper, abstract_only, api_key)`
  - LLM generates 2 MCQs + 1 flashcard per paper
  - MCQ: question + 4 options + answer key
  - Flashcard: Q|A pair
  - Returns `Card` objects

**Frontend: `frontend/src/components/Study.jsx`**
- Paper selection grid
- Active study session display
- Flashcard with flip animation
- MCQ with option buttons
- Progress tracking (correct/incorrect/total)
- Session completion summary with score

### API Endpoints
```
POST /api/study/generate
Request: {"paper_id": "2024.01.001", "api_key": "..."}
Response: {"cards": [{...}], "paper_id": "..."}

POST /api/study/review
Request: {"card_id": "...", "correct": true, "quality": 4, "paper_id": "..."}
Response: {"status": "success"}

GET /api/study/due-cards?paper_id=...
Response: {"cards": [...], "total": 5}
```

---

## Cross-Cutting Requirements

### 1. Reward Logging
All features log events to `data/reward_log.jsonl`:
```python
log_reward_event('feature_name', {'key': 'value'})
```

Examples:
- `trends_viewed`: `{'trends_count': 10}`
- `ideas_generated`: `{'topic': '...', 'ideas_count': 3}`
- `citations_suggested`: `{'papers_found': 5, 'citations': 2}`
- `papers_compared`: `{'count': 2}`
- `cards_generated`: `{'paper_id': '...', 'count': 3}`
- `card_reviewed`: `{'paper_id': '...', 'correct': true, 'quality': 4}`

### 2. LLM Router
All LLM calls use the hybrid Gemini/Groq router:
```python
from core.llm_helper import call_llm_with_retry

response = call_llm_with_retry(prompt, api_key, use_groq=False)
```

### 3. Tab Integration
Each feature is a top-level tab in `App.jsx` with icons:
- Trends: `<BarChart3>`
- Ideas: `<Sparkles>`
- Write (Citations): `<BookOpen>`
- Compare: `<Columns3>`
- Study: `<BookMarked>`

### 4. Data Persistence
- Ratings: `data/ratings.json`
- Collections: `data/collections.json`
- Study cards: `data/study.jsonl` (per-line JSON)
- Reward log: `data/reward_log.jsonl` (per-line JSON)

---

## API Reference

### Base URL
```
http://127.0.0.1:5001
```

### New Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/trends` | Get research trends |
| POST | `/api/ideas` | Find research gaps |
| POST | `/api/cite-suggest` | Suggest citations |
| POST | `/api/compare` | Compare papers |
| POST | `/api/study/generate` | Generate study cards |
| POST | `/api/study/review` | Log card review |
| GET | `/api/study/due-cards` | Get cards due for review |

---

## Frontend Integration

### Component Props
- `geminiApiKey: string` - API key for LLM calls
- `papers: array` - Paper database (for Compare, Study)
- `ratings: object` - User ratings (for filtering)

### Navigation
New tabs appear in header nav bar with icons and labels:
```jsx
<button className="nav-tab" onClick={() => setActiveTab('trends')}>
  <BarChart3 size={16} /> Trends
</button>
```

### Styling
All components use:
- CSS variables: `--accent-color`, `--text-secondary`, `--font-heading`
- Glass morphism: `border: rgba(255,255,255,0.1)`, `background: rgba(0,0,0,0.3)`
- Lucide React icons

---

## Testing

Run tests:
```bash
python -m pytest backend/tests/test_trends.py
python -m pytest backend/tests/test_new_features.py
```

---

## Performance Notes

- **Trends**: O(n) where n = number of papers
- **Idea Finder**: O(n log n) for sparse region detection
- **Citations**: O(m*n) where m = sentences, n = papers (can be optimized with HNSW)
- **Comparison**: Fixed cost (3 LLM calls per comparison)
- **Study**: O(k) for card generation where k = cards per paper (~3)

---

## Future Extensions

1. **Trends**: Real-time streaming updates via WebSocket
2. **Idea Finder**: Interactive 2D visualization with drag-to-explore
3. **Citations**: PDF extraction for full text analysis
4. **Comparison**: Custom field selection by user
5. **Study**: Mobile app with offline card storage and sync

---

## Troubleshooting

### Issue: LLM calls failing
- Check API key configuration
- Verify backend server running on port 5001
- Check network connectivity

### Issue: No trends detected
- Ensure papers have `published_date` field
- Verify date format: ISO 8601 (YYYY-MM-DD)

### Issue: Study cards not generating
- Check paper has non-empty abstract
- Verify LLM API credentials
- Check `data/study.jsonl` permissions

---

## References

- SM-2 Algorithm: https://en.wikipedia.org/wiki/Spaced_repetition#SM-2
- Thompson Sampling: https://en.wikipedia.org/wiki/Thompson_sampling
- BibTeX Format: https://en.wikipedia.org/wiki/BibTeX
- TF-IDF Cosine Similarity: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
