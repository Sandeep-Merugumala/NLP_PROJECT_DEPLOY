"""
Semantic embeddings for the recommender.

The base recommender trains on 256-dim TF-IDF vectors (lexical). Richer
Gemini semantic embeddings ("gemini-embedding-001", 768-dim) usually give a
noticeably better ranking because the model captures meaning, not just word
overlap. We compute them once, cache to disk, and attach them to papers as
`semantic_embedding`; the recommender uses them via feature_key.

Everything degrades gracefully: if the API is unavailable/rate-limited, we keep
whatever is already cached and the recommender falls back to TF-IDF.
"""

import os
import json
import time
from .llm_helper import get_gemini_embeddings_batch

SEMANTIC_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "semantic_embeddings.json")


def load_semantic_cache(path=SEMANTIC_FILE):
    """Return {paper_id: [floats]} from disk, or {} if none/unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_semantic_cache(cache, path=SEMANTIC_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def attach_semantic(papers, cache=None):
    """Attach `semantic_embedding` to each paper present in the cache.
    Returns the number of papers covered."""
    if cache is None:
        cache = load_semantic_cache()
    covered = 0
    for p in papers:
        vec = cache.get(p["id"])
        if vec:
            p["semantic_embedding"] = vec
            covered += 1
    return covered


def build_semantic_cache(papers, api_key, batch_size=100, max_papers=None):
    """
    Compute Gemini embeddings for papers not already cached, saving incrementally.
    Returns {"built": n_new, "covered": total_cached, "total": len(papers)}.
    Stops early (keeping progress) if the API starts failing.
    """
    cache = load_semantic_cache()
    todo = [p for p in papers if p["id"] not in cache]
    if max_papers:
        todo = todo[:max_papers]

    built = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        texts = [f"{p.get('title', '')}. {p.get('abstract', '')}"[:2000] for p in batch]
        # Retry the batch a few times with back-off (the embedding API rate-limits).
        vecs = None
        for attempt in range(4):
            vecs = get_gemini_embeddings_batch(texts, api_key)
            if vecs and len(vecs) == len(batch):
                break
            time.sleep(5 * (attempt + 1))  # 5s, 10s, 15s, 20s
            vecs = None
        if not vecs:
            # Still failing after retries — persist progress and stop (resumable).
            break
        for p, v in zip(batch, vecs):
            cache[p["id"]] = v
            built += 1
        save_semantic_cache(cache)
        time.sleep(1.5)  # gentle pacing between successful batches

    covered = sum(1 for p in papers if p["id"] in cache)
    return {"built": built, "covered": covered, "total": len(papers)}
