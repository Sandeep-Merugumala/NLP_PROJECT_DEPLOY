import os
import json
import random
import datetime
import numpy as np
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pypdf import PdfReader

# --- Sentence-Transformers for semantic notebook search ---
try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT_MODEL: _ST | None = None  # lazy singleton
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    _SBERT_MODEL = None


def _get_sbert_model():
    """Lazy-load and cache the SBERT model (loads once per process)."""
    global _SBERT_MODEL
    if not _SBERT_AVAILABLE:
        return None
    if _SBERT_MODEL is None:
        _SBERT_MODEL = _ST("all-MiniLM-L6-v2")
    return _SBERT_MODEL

from core.data_loader import load_or_process_papers, DATA_FILE
from core.recommender import train_recommender
from core.semantic import load_semantic_cache, attach_semantic, build_semantic_cache
from core.bandit import rank_with_bandit
from core.llm_helper import explain_recommendation, answer_pdf_question
from core.mmr import mmr_rerank
from core.trends import compute_trends, generate_trend_summary
from core.gap_finder import find_gap_for_topic
from core.citation_finder import process_citation_request
from core.study import generate_cards_from_paper, load_cards, review_card, save_cards, SM2Scheduler, RLCardScheduler
from core.semantic_grader import grade_answer
from core.llm_helper import call_llm_with_retry
from agents.compare_agent import create_comparison_agent

# --- Part 2 feature modules ---
from core.learning_path import LearningPathGenerator
from core.notebook import NotebookManager
from core.author_network import AuthorVenueNetwork
from core.watchlists import WatchlistManager
from agents.content_studio import PaperContentStudio, ContentFormat
from dataclasses import asdict as _dc_asdict

app = Flask(__name__)
CORS(app) # Enable CORS for frontend integration

RATINGS_FILE = os.path.join(os.path.dirname(__file__), "data", "ratings.json")
COLLECTIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "collections.json")
REWARD_LOG_FILE = os.path.join(os.path.dirname(__file__), "data", "reward_log.jsonl")

# Global in-memory variables
papers_db = []
ratings_db = {}
collections_db = {}
tfidf_vectorizer = None
tfidf_matrix = None
paper_index_map = {}   # paper_id -> row index in tfidf_matrix
semantic_vectors = {}  # paper_id -> Gemini semantic embedding (loaded from cache)
USE_SEMANTIC = False   # True once the semantic cache covers (almost) the whole corpus

def init_db():
    global papers_db, ratings_db, collections_db, tfidf_vectorizer, tfidf_matrix, paper_index_map
    
    # 1. Load papers (crawls arXiv if papers.json doesn't exist)
    papers_db = load_or_process_papers()
    
    # 2. Load ratings
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, "r") as f:
                ratings_db = json.load(f)
        except Exception:
            ratings_db = {}
    else:
        ratings_db = {}

    # Prune stale ratings whose paper is no longer in the corpus, so the header
    # counts stay consistent with what the digest/collections can actually show.
    _valid_ids = {p["id"] for p in papers_db}
    _stale = [pid for pid in ratings_db if pid not in _valid_ids]
    if _stale:
        for pid in _stale:
            ratings_db.pop(pid, None)
        try:
            with open(RATINGS_FILE, "w") as f:
                json.dump(ratings_db, f, indent=2)
        except Exception:
            pass

    # 3. Load collections
    if os.path.exists(COLLECTIONS_FILE):
        try:
            with open(COLLECTIONS_FILE, "r") as f:
                collections_db = json.load(f)
        except Exception:
            collections_db = {"My Library": []}
    else:
        collections_db = {"My Library": []}
        
    # 4. Fit Global TF-IDF Vectorizer for semantic search and collection expansions
    corpus = [f"{p['title']} [SEP] {p['abstract']}" for p in papers_db]
    tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # 5. Build paper_id → row index map (used by MMR and /api/similar)
    paper_index_map = {p["id"]: idx for idx, p in enumerate(papers_db)}

    # 5b. Load cached Gemini semantic embeddings (if built). Use them for the
    #     recommender/bandit when they cover (almost) the whole corpus; otherwise
    #     fall back to the 256-dim TF-IDF vectors.
    _load_semantic_state()

    # 6. Pre-warm the SBERT model so notebook notes are always encoded at
    #    384-dim (avoids 256-dim TF-IDF fallback on first note creation).
    if _SBERT_AVAILABLE:
        print("Pre-loading SBERT model (all-MiniLM-L6-v2)...")
        _get_sbert_model()
        print("SBERT model ready.")

    # 7. Train recommender on startup if ratings exist
    if ratings_db:
        print(f"Training recommender on startup (semantic={USE_SEMANTIC})...")
        papers_db = train_recommender(papers_db, ratings_db, vectors=_rec_vectors())
        save_papers_to_disk()
    else:
        # Reset relevance scores if no ratings are found (e.g., after clearing data)
        for p in papers_db:
            p["relevance_score"] = 0.0
        save_papers_to_disk()


def _load_semantic_state():
    """Load the semantic cache and decide whether to use it, based on coverage."""
    global semantic_vectors, USE_SEMANTIC
    semantic_vectors = load_semantic_cache()
    covered = sum(1 for p in papers_db if p["id"] in semantic_vectors)
    total = max(1, len(papers_db))
    USE_SEMANTIC = covered / total >= 0.9
    print(f"Semantic coverage: {covered}/{total} -> semantic={USE_SEMANTIC}")


def _rec_vectors():
    """Vectors dict to feed the recommender/bandit, or None to use TF-IDF."""
    return semantic_vectors if USE_SEMANTIC else None

def save_papers_to_disk():
    global papers_db
    # Keep the big semantic vectors out of papers.json (they live in the
    # semantic_embeddings cache); strip them from the serialized copy.
    with open(DATA_FILE, "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "semantic_embedding"} for p in papers_db], f, indent=2)

def save_ratings():
    global ratings_db
    with open(RATINGS_FILE, "w") as f:
        json.dump(ratings_db, f, indent=2)

def save_collections():
    global collections_db
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(collections_db, f, indent=2)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NOTEBOOK_FILE = os.path.join(DATA_DIR, "notebook.jsonl")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlists.json")


# --- Lightweight adapters so the Part 2 modules (which expect injected
#     embeddings/LLM/RAG dependencies) run on the app's existing assets. ---
class _SemanticEmbedder:
    """Semantic embedder using sentence-transformers (all-MiniLM-L6-v2).

    Produces L2-normalised 384-dim vectors so cosine similarity ==
    np.dot(a, b).  Falls back to the TF-IDF vectorizer when the model
    is unavailable (e.g. import error).
    """
    def encode(self, text: str) -> np.ndarray:
        model = _get_sbert_model()
        if model is not None:
            vec = model.encode(text or "", normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        # Graceful fallback to TF-IDF
        if tfidf_vectorizer is not None:
            return tfidf_vectorizer.transform([text or ""]).toarray()[0]
        return np.zeros(256, dtype=np.float32)


# Keep old name as alias so any other code that still references _TfidfEmbedder works
_TfidfEmbedder = _SemanticEmbedder


class _LLMRouter:
    """Routes .generate(prompt, ...) through the Gemini/Groq hybrid helper."""
    def __init__(self, api_key=""):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def generate(self, prompt, temperature=0.5, max_tokens=800, **kwargs):
        # Accept extra generation kwargs that some Part 2 modules pass.
        text = call_llm_with_retry(prompt, self.api_key) or ""
        # When the caller wants JSON, strip any ```json fences / prose so
        # json.loads() in the module succeeds instead of falling back to empty.
        if kwargs.get("format_json") and text:
            import re
            m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if m:
                return m.group(1)
        return text


class _AbstractRAG:
    """Minimal RAG that serves each paper's abstract as retrievable chunks."""
    def retrieve(self, paper_id, top_k=5):
        paper = next((p for p in papers_db if p.get("id") == paper_id), None)
        if not paper:
            return []
        abstract = paper.get("abstract", "") or paper.get("title", "")
        # Split the abstract into a few pseudo-chunks so top_k has something to work with
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if s.strip()]
        if not sentences:
            sentences = [abstract]
        chunks, step = [], max(1, len(sentences) // max(1, top_k))
        for i in range(0, len(sentences), step):
            text = " ".join(sentences[i:i + step])
            chunks.append({"content": text, "text": text, "page_idx": 0})
        return chunks[:top_k] or [{"content": abstract, "text": abstract, "page_idx": 0}]


def append_reward_event(event_type, paper_id, value=None):
    """Append a single reward/interaction event to the JSONL log."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "paper_id": paper_id,
        "value": value
    }
    try:
        with open(REWARD_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Warning: could not write reward log: {e}")

@app.route("/api/papers", methods=["GET"])
def get_papers():
    return jsonify({
        "papers": papers_db,
        "ratings": ratings_db
    })

@app.route("/api/rate", methods=["POST"])
def rate_paper():
    global papers_db, ratings_db
    data = request.json
    paper_id = data.get("paper_id")
    rating = data.get("rating") # 1, -1, or 0 (to clear)
    
    if not paper_id:
        return jsonify({"error": "Missing paper_id"}), 400
        
    # Update ratings
    if rating == 0:
        if paper_id in ratings_db:
            del ratings_db[paper_id]
    else:
        ratings_db[paper_id] = rating
        
    save_ratings()
    
    # Log reward event for future RL / evaluation use
    event_label = "upvote" if rating == 1 else ("downvote" if rating == -1 else "clear")
    append_reward_event(event_label, paper_id, rating)
    
    # Retrain recommender
    papers_db = train_recommender(papers_db, ratings_db, vectors=_rec_vectors())
    save_papers_to_disk()
    
    return jsonify({
        "status": "success",
        "papers": papers_db,
        "ratings": ratings_db
    })

@app.route("/api/digest", methods=["GET"])
def get_digest():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    
    # diversity λ: 1.0 = pure relevance, 0.0 = maximum diversity
    try:
        lam = float(request.args.get("diversity", "1.0"))
        lam = max(0.0, min(1.0, lam))
    except ValueError:
        lam = 1.0

    # Filter by date range first
    filtered_db = papers_db
    if start_date:
        filtered_db = [p for p in filtered_db if p["published"] >= start_date]
    if end_date:
        filtered_db = [p for p in filtered_db if p["published"] <= end_date]

    # Slice recent papers based on range
    sorted_papers = sorted(filtered_db, key=lambda p: p["published"], reverse=True)
    candidates = sorted_papers[:40]

    # Always surface the user's liked papers in the digest, even if they're not
    # among the most recent (so the "liked" count the header shows is reflected here).
    liked_ids = {pid for pid, r in ratings_db.items() if r == 1}
    if liked_ids:
        have = {p["id"] for p in candidates}
        for p in filtered_db:
            if p["id"] in liked_ids and p["id"] not in have:
                candidates.append(p)

    # Ordering: default = pure relevance (exploit). With ?explore=1, use the
    # LinUCB contextual bandit so uncertain-but-promising papers get surfaced
    # (explore/exploit trade-off — the RL layer on top of the LR scores).
    explore = str(request.args.get("explore", "")).lower() in ("1", "true", "yes")
    if explore:
        candidates_by_rel = rank_with_bandit(
            candidates, papers_db, ratings_db, vectors=_rec_vectors(), alpha=0.6
        )
    else:
        candidates_by_rel = sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)

    # Apply MMR re-ranking if diversity is requested
    tfidf_dense = tfidf_matrix.toarray()
    digest_papers = mmr_rerank(
        candidates_by_rel,
        tfidf_dense,
        paper_index_map,
        lam=lam,
        top_k=15
    )

    # Compute interest clusters from upvoted papers to surface in UI
    interests = _compute_interests()

    return jsonify({
        "papers": digest_papers,
        "interests": interests,
        "diversity_lam": lam,
        "explore": explore,
        "ranker": ("semantic+LR" if USE_SEMANTIC else "tfidf+LR") + (" + bandit" if explore else ""),
    })

@app.route("/api/search", methods=["POST"])
def search_papers():
    global tfidf_vectorizer, tfidf_matrix
    body = request.json or {}
    query = body.get("query", "")
    expand = body.get("expand", False)  # Feature G: query expansion
    
    # Feature G: expand query with top terms from upvoted papers
    if expand and ratings_db:
        upvoted_ids = [pid for pid, r in ratings_db.items() if r == 1]
        if upvoted_ids:
            upvoted_indices = [
                paper_index_map[pid] for pid in upvoted_ids
                if pid in paper_index_map
            ]
            if upvoted_indices:
                avg_vec = tfidf_matrix.toarray()[upvoted_indices].mean(axis=0)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                top_term_indices = np.argsort(avg_vec)[::-1][:5]
                expansion_terms = " ".join([feature_names[i] for i in top_term_indices])
                query = query + " " + expansion_terms if query.strip() else expansion_terms

    if not query.strip():
        # Return top relevance papers by default
        results = sorted(papers_db, key=lambda p: p["relevance_score"], reverse=True)
        return jsonify({"results": results[:20], "expanded_query": query})
        
    # Compute query vector
    query_vec = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarities
    similarities = np.dot(tfidf_matrix.toarray(), query_vec.T.toarray()).flatten()
    
    # Combine TF-IDF similarity with personal relevance
    search_results = []
    for idx, paper in enumerate(papers_db):
        sim = float(similarities[idx])
        # Score combines similarity (primary) and user relevance (secondary)
        combined_score = sim * 100.0 + (paper["relevance_score"] * 0.1)
        
        search_results.append({
            "paper": paper,
            "similarity": sim,
            "combined_score": combined_score
        })
        
    # Sort by combined score descending
    search_results = sorted(search_results, key=lambda x: x["combined_score"], reverse=True)
    
    # Format return
    results = [item["paper"] for item in search_results if item["similarity"] > 0]
    
    # If no keywords matched, do a basic text search fallback using word boundaries
    if not results:
        import re
        query_words = [w.lower() for w in query.split() if w]
        for paper in papers_db:
            title_abs = (paper["title"] + " " + paper["abstract"]).lower()
            match = True
            for qw in query_words:
                if not re.search(r'\b' + re.escape(qw), title_abs):
                    match = False
                    break
            if match:
                results.append(paper)
                
    return jsonify({"results": results[:20], "expanded_query": query})

@app.route("/api/active-learning", methods=["GET"])
def get_active_learning():
    """
    Stratified sampling with K-Means diversity clustering:
    Filters the top 20 unrated papers nearest the decision boundary, clusters their t-SNE coordinates
    into 5 distinct groups, and selects the paper closest to the centroid of each group.
    """
    unrated = [p for p in papers_db if p["id"] not in ratings_db]
    if not unrated:
        return jsonify({"papers": []})
        
    boundary_papers = sorted(unrated, key=lambda p: abs(p["relevance_score"]))
    if len(boundary_papers) <= 5:
        return jsonify({"papers": boundary_papers})
        
    # Apply K-Means clustering on t-SNE coordinates for the top 20 papers
    candidates = boundary_papers[:20]
    coords = np.array([[p["x"], p["y"]] for p in candidates])
    
    try:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        selected_papers = []
        for cluster_idx in range(5):
            cluster_mask = (labels == cluster_idx)
            cluster_candidates = [candidates[i] for i, m in enumerate(cluster_mask) if m]
            if cluster_candidates:
                centroid = kmeans.cluster_centers_[cluster_idx]
                # Sort by squared distance to centroid
                cluster_candidates = sorted(cluster_candidates, key=lambda p: (p["x"] - centroid[0])**2 + (p["y"] - centroid[1])**2)
                selected_papers.append(cluster_candidates[0])
                
        # Fallback padding if we failed to fill 5 clusters
        for p in boundary_papers:
            if len(selected_papers) >= 5:
                break
            if p not in selected_papers:
                selected_papers.append(p)
    except Exception as e:
        print(f"K-Means clustering failed: {e}. Falling back to standard boundary papers.")
        selected_papers = boundary_papers[:5]
        
    return jsonify({
        "papers": selected_papers[:5]
    })

@app.route("/api/chat-pdf", methods=["POST"])
def chat_pdf():
    data = request.json
    paper_id = data.get("paper_id")
    question = data.get("question")
    api_key = data.get("api_key")
    
    if not paper_id or not question:
        return jsonify({"error": "Missing paper_id or question"}), 400
        
    # PDF Cache directory Setup
    pdf_cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
    os.makedirs(pdf_cache_dir, exist_ok=True)
    cache_file = os.path.join(pdf_cache_dir, f"{paper_id}.json")
    
    chunks = []
    
    # 1. Load from cache or download and split PDF
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"Failed to read cache: {e}")
            
    if not chunks:
        temp_pdf_path = os.path.join(pdf_cache_dir, f"{paper_id}.pdf")
        arxiv_pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        # Download paper PDF
        try:
            print(f"Downloading PDF from {arxiv_pdf_url}...")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(arxiv_pdf_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(temp_pdf_path, "wb") as out_file:
                out_file.write(response.read())
        except Exception as e:
            return jsonify({"error": f"Failed to download PDF from arXiv: {e}"}), 500
            
        # Parse PDF text page-by-page (Hierarchical Parent-Child Chunking)
        try:
            reader = PdfReader(temp_pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # 1. Create Parent chunks (size 1500, overlap 300)
                parent_limit = 1500
                parent_overlap = 300
                parent_start = 0
                parents = []
                while parent_start < len(text):
                    parent_end = parent_start + parent_limit
                    parents.append(text[parent_start:parent_end])
                    parent_start += (parent_limit - parent_overlap)
                
                # 2. Create Child chunks (size 300, overlap 50) and link them to parent context
                child_limit = 300
                child_overlap = 50
                child_start = 0
                while child_start < len(text):
                    child_end = child_start + child_limit
                    child_text = text[child_start:child_end]
                    
                    # Find which parent chunk contains this child chunk
                    parent_text = ""
                    for p in parents:
                        if child_text in p or p in child_text:
                            parent_text = p
                            break
                    if not parent_text and parents:
                        parent_text = parents[0]
                        
                    chunks.append({
                        "child_text": child_text,
                        "text": parent_text if parent_text else child_text, # Parent context sent to LLM
                        "page_idx": page_idx + 1
                    })
                    child_start += (child_limit - child_overlap)
            
            # Save parsed chunks as cache
            with open(cache_file, "w") as f:
                json.dump(chunks, f, indent=2)
                
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF text: {e}"}), 500
        finally:
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
    if not chunks:
        return jsonify({"error": "No text content could be extracted from this PDF"}), 500
        
    # 2. Vectorize child chunks and question to find top matching chunks via Hybrid Semantic-Lexical Search
    top_chunks = []
    
    use_semantic = api_key and not api_key.startswith("gsk_")
    if use_semantic:
        try:
            from core.llm_helper import get_gemini_embedding, get_gemini_embeddings_batch
            
            # Embed uncached chunks in batches of 50 to minimize network roundtrips
            uncached_chunks = [c for c in chunks if "embedding" not in c]
            if uncached_chunks:
                print(f"Generating dense vector embeddings for {len(uncached_chunks)} uncached PDF page chunks...")
                texts_to_embed = [c.get("child_text", c["text"]) for c in uncached_chunks]
                batch_limit = 50
                for start_idx in range(0, len(texts_to_embed), batch_limit):
                    subset = texts_to_embed[start_idx : start_idx + batch_limit]
                    subset_embeddings = get_gemini_embeddings_batch(subset, api_key)
                    if subset_embeddings:
                        for offset, emb in enumerate(subset_embeddings):
                            uncached_chunks[start_idx + offset]["embedding"] = emb
                            
                # Save updated chunks back to disk cache so future chats do not recompute embeddings
                try:
                    with open(cache_file, "w") as f:
                        json.dump(chunks, f, indent=2)
                except Exception as cache_err:
                    print(f"Failed to update cache file with embeddings: {cache_err}")
            
            # Fetch user question embedding vector
            q_emb = get_gemini_embedding(question, api_key)
            
            if q_emb and all("embedding" in c for c in chunks):
                q_vec = np.array(q_emb)
                q_norm = np.linalg.norm(q_vec)
                
                # Compute Cosine Similarity between query embedding and cached chunk embeddings
                for c in chunks:
                    c_vec = np.array(c["embedding"])
                    c_norm = np.linalg.norm(c_vec)
                    if q_norm > 0 and c_norm > 0:
                        sim = float(np.dot(c_vec, q_vec) / (c_norm * q_norm))
                    else:
                        sim = 0.0
                    c["score"] = sim
                    
                ranked_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
                top_chunks = ranked_chunks[:3]
                print(f"Semantic dense search completed successfully. Top chunk score: {top_chunks[0]['score']:.4f}")
        except Exception as semantic_err:
            print(f"Semantic embedding search failed: {semantic_err}. Falling back to TF-IDF.")
            
    if not top_chunks:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus = [c.get("child_text", c["text"]) for c in chunks]
            vectorizer.fit(corpus + [question])
            
            chunk_vectors = vectorizer.transform(corpus)
            question_vector = vectorizer.transform([question])
            
            similarities = np.dot(chunk_vectors.toarray(), question_vector.T.toarray()).flatten()
            
            for idx, sim in enumerate(similarities):
                chunks[idx]["score"] = float(sim)
                
            ranked_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
            top_chunks = ranked_chunks[:3]
            print("TF-IDF lexical search fallback completed successfully.")
        except Exception as tfidf_err:
            print(f"TF-IDF chunk similarity failed: {tfidf_err}. Defaulting to first 3 page chunks.")
            top_chunks = chunks[:3]
            for c in top_chunks:
                c["score"] = 0.0
            
    # 3. Ask Gemini/Heuristic to answer the question based on context chunks
    answer = answer_pdf_question(top_chunks, question, api_key)
    
    return jsonify({
        "answer": answer,
        "retrieved_chunks": top_chunks
    })

@app.route("/api/explain", methods=["POST"])
def explain_paper():
    data = request.json
    paper_id = data.get("paper_id")
    api_key = data.get("api_key") # Can be passed dynamically from frontend
    
    if not paper_id:
        return jsonify({"error": "Missing paper_id"}), 400
        
    # Find target paper
    target_paper = next((p for p in papers_db if p["id"] == paper_id), None)
    if not target_paper:
        return jsonify({"error": "Paper not found"}), 404
        
    # Gather titles of upvoted papers
    upvoted_titles = []
    for pid, rating in ratings_db.items():
        if rating == 1:
            upvoted_p = next((p for p in papers_db if p["id"] == pid), None)
            if upvoted_p:
                upvoted_titles.append(upvoted_p["title"])
                
    # Call helper
    analysis = explain_recommendation(
        upvoted_titles,
        target_paper["title"],
        target_paper["abstract"],
        target_paper["category_name"],
        api_key=api_key
    )
    
    return jsonify(analysis)

@app.route("/api/collections", methods=["GET", "POST"])
def manage_collections():
    global collections_db
    
    if request.method == "GET":
        return jsonify({"collections": collections_db})
        
    data = request.json
    action = data.get("action") # "create", "add", "remove"
    name = data.get("name")
    paper_id = data.get("paper_id")
    
    if not name:
        return jsonify({"error": "Missing collection name"}), 400
        
    if action == "create":
        if name not in collections_db:
            collections_db[name] = []
            save_collections()
            
    elif action == "add":
        if name in collections_db and paper_id and paper_id not in collections_db[name]:
            collections_db[name].append(paper_id)
            save_collections()
            
    elif action == "remove":
        if name in collections_db and paper_id and paper_id in collections_db[name]:
            collections_db[name].remove(paper_id)
            save_collections()
            
    return jsonify({
        "status": "success",
        "collections": collections_db
    })

@app.route("/api/collections/<name>/recommendations", methods=["GET"])
def get_collection_recommendations(name):
    global tfidf_matrix, tfidf_vectorizer
    
    if name not in collections_db or not collections_db[name]:
        return jsonify({"papers": []})
        
    # Get paper IDs in collection
    col_paper_ids = set(collections_db[name])
    
    # Find indices of collection papers in global papers list
    indices = [idx for idx, p in enumerate(papers_db) if p["id"] in col_paper_ids]
    
    if not indices:
        return jsonify({"papers": []})
        
    # Average embedding of collection papers
    col_embeddings = tfidf_matrix.toarray()[indices]
    avg_embedding = np.mean(col_embeddings, axis=0)
    
    # Compute similarity against all papers
    similarities = np.dot(tfidf_matrix.toarray(), avg_embedding).flatten()
    
    recommendations = []
    for idx, paper in enumerate(papers_db):
        if paper["id"] in col_paper_ids:
            continue # Skip papers already in collection
            
        recommendations.append({
            "paper": paper,
            "similarity": float(similarities[idx])
        })
        
    # Sort by similarity descending
    recommendations = sorted(recommendations, key=lambda x: x["similarity"], reverse=True)
    
    return jsonify({
        "papers": [item["paper"] for item in recommendations[:5]]
    })

@app.route("/api/conference", methods=["GET"])
def get_conference_planner():
    """
    Generates a set of mock conference poster sessions.
    Ranks them based on the average relevance of papers in each session.
    """
    # Define 4 mock poster sessions
    sessions = [
        {
            "id": "session_1",
            "name": "Poster Session A: Deep Learning Foundations & Optimization",
            "time": "Day 1, 09:00 - 11:00",
            "keywords": ["gradient", "loss", "network", "train", "optimization", "transformer", "attention"]
        },
        {
            "id": "session_2",
            "name": "Poster Session B: Computer Vision & Generative Models",
            "time": "Day 1, 14:00 - 16:00",
            "keywords": ["image", "object", "segmentation", "detection", "diffusion", "gan", "cnn", "vision"]
        },
        {
            "id": "session_3",
            "name": "Poster Session C: Natural Language & Agentic LLMs",
            "time": "Day 2, 09:00 - 11:00",
            "keywords": ["text", "language", "nlp", "translation", "llm", "agent", "prompt", "reasoning"]
        },
        {
            "id": "session_4",
            "name": "Poster Session D: Reinforcement Learning & Controls",
            "time": "Day 2, 14:00 - 16:00",
            "keywords": ["policy", "agent", "reinforcement", "control", "rl", "robot", "action"]
        }
    ]
    
    # Distribute papers to sessions based on keywords matching
    session_papers = {s["id"]: [] for s in sessions}
    
    for paper in papers_db:
        content = (paper["title"] + " " + paper["abstract"]).lower()
        
        # Calculate matching scores for each session
        scores = {}
        for s in sessions:
            score = sum(1 for kw in s["keywords"] if kw in content)
            scores[s["id"]] = score
            
        # Assign to best matching session, or fallback based on primary category
        best_session_id = max(scores, key=scores.get)
        if scores[best_session_id] > 0:
            session_papers[best_session_id].append(paper)
        else:
            # Fallback mapping
            cat = paper["primary_category"]
            if cat == "cs.CV":
                session_papers["session_2"].append(paper)
            elif cat == "cs.CL":
                session_papers["session_3"].append(paper)
            elif cat == "cs.LG":
                session_papers["session_1"].append(paper)
            else:
                session_papers["session_4"].append(paper)
                
    # Rank sessions by average paper relevance score
    ranked_sessions = []
    for s in sessions:
        papers_in_s = session_papers[s["id"]]
        avg_relevance = 0.0
        if papers_in_s:
            avg_relevance = float(np.mean([p["relevance_score"] for p in papers_in_s]))
            
        # Sort papers in session by relevance score descending
        sorted_papers = sorted(papers_in_s, key=lambda p: p["relevance_score"], reverse=True)
        
        ranked_sessions.append({
            "id": s["id"],
            "name": s["name"],
            "time": s["time"],
            "avg_relevance": float(np.round(avg_relevance, 2)),
            "papers": sorted_papers
        })
        
    # Sort sessions by average relevance descending
    ranked_sessions = sorted(ranked_sessions, key=lambda s: s["avg_relevance"], reverse=True)
    
    return jsonify({
        "sessions": ranked_sessions
    })

@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    data = request.json
    query = data.get("query", "")
    api_key = data.get("api_key") or os.environ.get("GEMINI_API_KEY")
    
    if not query.strip():
        return jsonify({"error": "Query cannot be empty"}), 400
        
    try:
        from core.agent_orchestrator import run_agent_workflow
        output = run_agent_workflow(query, api_key)
        return jsonify({
            "report": output.get("final_report", ""),
            "logs": output.get("agent_logs", []),
            "papers": output.get("papers", [])
        })
    except Exception as e:
        print(f"Error running agent workflow: {e}")
        return jsonify({"error": f"Failed to execute Multi-Agent workflow: {e}"}), 500


# ─────────────────────────────────────────────────────────────
# Feature B: Similar Papers  GET /api/similar/<paper_id>
# ─────────────────────────────────────────────────────────────
@app.route("/api/similar/<path:paper_id>", methods=["GET"])
def get_similar_papers(paper_id):
    """Return top-5 similar papers using cosine similarity on the TF-IDF matrix."""
    if paper_id not in paper_index_map:
        return jsonify({"error": "Paper not found"}), 404

    target_idx = paper_index_map[paper_id]
    dense = tfidf_matrix.toarray()
    target_vec = dense[target_idx]
    target_norm = np.linalg.norm(target_vec)

    similarities = []
    for pid, idx in paper_index_map.items():
        if pid == paper_id:
            continue
        vec = dense[idx]
        norm = np.linalg.norm(vec)
        sim = float(np.dot(target_vec, vec) / (target_norm * norm + 1e-9))
        similarities.append((pid, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_ids = [pid for pid, _ in similarities[:5]]
    top_sims = {pid: sim for pid, sim in similarities[:5]}

    similar_papers = [
        {**p, "similarity": round(top_sims[p["id"]], 4)}
        for p in papers_db if p["id"] in top_ids
    ]
    similar_papers.sort(key=lambda p: p["similarity"], reverse=True)

    return jsonify({"papers": similar_papers})


# ─────────────────────────────────────────────────────────────
# Feature D helper + endpoint: Interest Clusters  GET /api/interests
# ─────────────────────────────────────────────────────────────
def _compute_interests():
    """K-Means cluster upvoted paper embeddings and return topic descriptors."""
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

    # Identify top TF-IDF terms for each cluster centroid
    feature_names = tfidf_vectorizer.get_feature_names_out()
    interests = []
    for ci in range(n_clusters):
        cluster_papers = [upvoted[i] for i, lbl in enumerate(labels) if lbl == ci]
        if not cluster_papers:
            continue
        # Average TF-IDF vector for papers in this cluster
        indices = [paper_index_map[p["id"]] for p in cluster_papers if p["id"] in paper_index_map]
        if not indices:
            continue
        avg_vec = tfidf_matrix.toarray()[indices].mean(axis=0)
        top_term_indices = np.argsort(avg_vec)[::-1][:4]
        top_terms = [feature_names[i] for i in top_term_indices]
        # Representative paper title
        rep_paper = cluster_papers[0]["title"]
        interests.append({
            "cluster": ci,
            "label": f"Topic {ci + 1}: {', '.join(top_terms)}",
            "top_terms": top_terms,
            "count": len(cluster_papers),
            "representative": rep_paper
        })

    return interests


@app.route("/api/interests", methods=["GET"])
def get_interests():
    """Return interest cluster descriptors derived from upvoted papers."""
    return jsonify({"interests": _compute_interests()})


# ─────────────────────────────────────────────────────────────
# Feature A: Live Stats  GET /api/stats
# ─────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Aggregate counts and activity stats for the sidebar dashboard."""
    total = len(papers_db)
    upvoted = sum(1 for r in ratings_db.values() if r == 1)
    downvoted = sum(1 for r in ratings_db.values() if r == -1)
    bookmarked = len(collections_db.get("My Library", []))

    # Top category among upvoted papers
    top_category = None
    if ratings_db:
        upvoted_papers = [p for p in papers_db if ratings_db.get(p["id"]) == 1]
        if upvoted_papers:
            from collections import Counter
            cat_counts = Counter(p.get("category_name", "Unknown") for p in upvoted_papers)
            top_category = cat_counts.most_common(1)[0][0]

    # Recent rating activity (last 7 events from reward_log)
    recent_events = []
    if os.path.exists(REWARD_LOG_FILE):
        try:
            with open(REWARD_LOG_FILE, "r") as f:
                lines = f.readlines()
            for line in reversed(lines[-20:]):
                entry = json.loads(line.strip())
                recent_events.append(entry)
                if len(recent_events) >= 7:
                    break
            recent_events.reverse()
        except Exception:
            pass

    return jsonify({
        "total_papers": total,
        "upvoted": upvoted,
        "downvoted": downvoted,
        "bookmarked": bookmarked,
        "top_category": top_category,
        "recent_events": recent_events
    })


# ─────────────────────────────────────────────────────────────
# Feature F: Reward Event Logging  POST /api/log-event
# ─────────────────────────────────────────────────────────────
@app.route("/api/log-event", methods=["POST"])
def log_event():
    """Log any user interaction event (bookmark, click, etc.)."""
    data = request.json or {}
    event_type = data.get("event_type")
    paper_id = data.get("paper_id")
    value = data.get("value")

    if not event_type or not paper_id:
        return jsonify({"error": "Missing event_type or paper_id"}), 400

    append_reward_event(event_type, paper_id, value)
    return jsonify({"status": "logged"})


# ─────────────────────────────────────────────────────────────
# Feature F: Reading History  GET /api/history
# ─────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    """Return chronological list of reward/interaction events with paper metadata."""
    limit = int(request.args.get("limit", 50))
    events = []

    if os.path.exists(REWARD_LOG_FILE):
        try:
            with open(REWARD_LOG_FILE, "r") as f:
                lines = f.readlines()
            for line in reversed(lines):
                try:
                    entry = json.loads(line.strip())
                    # Attach paper title for display
                    paper = next((p for p in papers_db if p["id"] == entry["paper_id"]), None)
                    if paper:
                        entry["paper_title"] = paper["title"]
                        entry["paper_authors"] = paper.get("authors", [])[:2]
                        entry["category_name"] = paper.get("category_name", "")
                    events.append(entry)
                    if len(events) >= limit:
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"History read error: {e}")

    return jsonify({"events": events})


# ============= NEW FEATURES: 5 CORE TABS =============

@app.route("/api/trends", methods=["GET"])
def api_trends():
    """Research Trends Explorer"""
    try:
        trends_data = compute_trends(papers_db)
        api_key = request.args.get('api_key', '')
        
        # Generate LLM summary
        summary = generate_trend_summary(trends_data, api_key) if api_key else ''
        trends_data['trend_summary'] = summary
        
        # Log event
        append_reward_event('trends_viewed', 'all', len(trends_data.get('rising_topics', [])))
        
        return jsonify(trends_data)
    except Exception as e:
        print(f"Trends error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/ideas", methods=["POST"])
def api_ideas():
    """Research Gap and Idea Finder"""
    try:
        data = request.get_json()
        topic_keyword = data.get('topic_keyword', '')
        api_key = data.get('api_key', '')
        
        if not topic_keyword:
            return jsonify({'error': 'Missing topic_keyword'}), 400
        
        # For now, use empty t-SNE coords (in production would come from ScholarMap)
        embeddings = tfidf_matrix.toarray() if tfidf_matrix is not None else np.array([])
        t_sne_coords = np.random.randn(len(papers_db), 2) if papers_db else np.array([])
        
        result = find_gap_for_topic(papers_db, embeddings, t_sne_coords, topic_keyword, api_key)
        
        append_reward_event('ideas_generated', topic_keyword, len(result.get('ideas', [])))
        
        return jsonify(result)
    except Exception as e:
        print(f"Ideas error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/cite-suggest", methods=["POST"])
def api_cite_suggest():
    """Citation Finder / Related Work Writer"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        api_key = data.get('api_key', '')
        
        if not text:
            return jsonify({'error': 'Missing text'}), 400
        
        result = process_citation_request(text, papers_db, tfidf_vectorizer, tfidf_matrix, api_key)
        
        stats = result.get('statistics', {})
        append_reward_event('citations_suggested', 'text', stats.get('papers_found', 0))
        
        return jsonify(result)
    except Exception as e:
        print(f"Citation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Paper Comparison Studio"""
    try:
        data = request.get_json()
        paper_ids = data.get('paper_ids', [])
        api_key = data.get('api_key', '')
        
        if not paper_ids or len(paper_ids) < 2:
            return jsonify({'error': 'Please select at least 2 papers'}), 400
        
        agent = create_comparison_agent(api_key, use_groq=False)
        result = agent.run(paper_ids, papers_db)
        
        append_reward_event('papers_compared', str(paper_ids), len(paper_ids))
        
        return jsonify(result)
    except Exception as e:
        print(f"Compare error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/study/generate", methods=["POST"])
def api_study_generate():
    """Generate study cards for a paper"""
    try:
        data = request.get_json()
        paper_id = data.get('paper_id', '')
        api_key = data.get('api_key', '')
        
        if not paper_id:
            return jsonify({'error': 'Missing paper_id'}), 400
        
        # Find paper
        paper = next((p for p in papers_db if p.get('id') == paper_id), None)
        if not paper:
            return jsonify({'error': 'Paper not found'}), 404
        
        # Generate cards (up to 10)
        cards = generate_cards_from_paper(paper, abstract_only=True, api_key=api_key)

        # Persist: merge the new cards into the study deck (dedupe by id) so that
        # spaced-repetition scheduling and the RL bandit have state to work with.
        existing = load_cards()
        by_id = {c.id: c for c in existing}
        for c in cards:
            by_id.setdefault(c.id, c)
        save_cards(list(by_id.values()))

        # Convert to dicts
        cards_data = [c.to_dict() for c in cards]

        append_reward_event('cards_generated', paper_id, len(cards))

        return jsonify({'cards': cards_data, 'paper_id': paper_id, 'count': len(cards_data)})
    except Exception as e:
        print(f"Study generate error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/study/review", methods=["POST"])
def api_study_review():
    """Log card review and update scheduling.
    
    Accepts either:
      - Traditional: {correct, quality} (manual self-grading)
      - NLP STS:     {typed_answer} (automatic semantic grading via SBERT)
    """
    try:
        data = request.get_json()
        card_id = data.get('card_id', '')
        paper_id = data.get('paper_id', '')
        typed_answer = data.get('typed_answer', '')

        # Load the deck
        all_cards = load_cards()
        target_card = None
        for c in all_cards:
            if c.id == card_id:
                target_card = c
                break

        # Determine quality: use STS auto-grading if typed_answer is provided
        sts_result = None
        if typed_answer and target_card:
            # ---- NLP Feature 1: Semantic Textual Similarity Grading ----
            sts_result = grade_answer(typed_answer, target_card.answer)
            quality = sts_result['quality']
            correct = sts_result['correct']
        else:
            correct = data.get('correct', False)
            quality = data.get('quality', 4 if correct else 1)

        # Update the reviewed card (SM-2 interval + Thompson Sampling)
        if target_card is not None:
            review_card(target_card, quality)
            save_cards(all_cards)

        # Log to reward_log
        append_reward_event('card_reviewed', paper_id or card_id, quality)

        # Use the RL bandit to recommend which card to study next
        pool = all_cards if not paper_id else [c for c in all_cards if c.paper_id == paper_id]
        next_card = RLCardScheduler.select_next_card([c for c in pool if c.id != card_id])

        response = {
            'status': 'success',
            'card': target_card.to_dict() if target_card else None,
            'next_card': next_card.to_dict() if next_card else None,
        }

        # Include STS grading details when available
        if sts_result:
            response['sts_grading'] = sts_result

        return jsonify(response)
    except Exception as e:
        print(f"Study review error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/study/due-cards", methods=["GET"])
def api_study_due_cards():
    """Get cards due for review"""
    try:
        paper_id = request.args.get('paper_id', '')
        
        # Load cards from file
        all_cards = load_cards()
        
        # Filter by paper if specified
        if paper_id:
            all_cards = [c for c in all_cards if c.paper_id == paper_id]
        
        # Get due cards
        due_cards = SM2Scheduler.get_due_cards(all_cards)
        due_data = [c.to_dict() for c in due_cards]
        
        return jsonify({'cards': due_data, 'total': len(due_cards)})
    except Exception as e:
        print(f"Due cards error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/study/grade", methods=["POST"])
def api_study_grade():
    """
    Grade a user's typed answer against the reference answer
    using Sentence-BERT Semantic Textual Similarity (STS).

    NLP Feature: Replaces manual self-grading with automatic
    cosine-similarity-based scoring using SBERT embeddings.

    Request body:
        {
            "user_answer": "The user's typed response",
            "reference_answer": "The correct answer text",
            "card_id": "optional card ID for context"
        }

    Response:
        {
            "similarity": 0.73,
            "quality": 3,
            "correct": true,
            "feedback": "...",
            "method": "sbert_sts"
        }
    """
    try:
        data = request.get_json()
        user_answer = data.get('user_answer', '')
        reference_answer = data.get('reference_answer', '')

        if not reference_answer:
            return jsonify({'error': 'Missing reference_answer'}), 400

        result = grade_answer(user_answer, reference_answer)
        return jsonify(result)
    except Exception as e:
        print(f"Study grade error: {e}")
        return jsonify({'error': str(e)}), 500


# ===================================================================
#  PART 2 FEATURES
# ===================================================================

@app.route("/api/learning-path", methods=["POST"])
def api_learning_path():
    """Build an ordered foundational→advanced reading path for a topic."""
    try:
        data = request.get_json() or {}
        topic = (data.get("topic") or "").strip()
        api_key = data.get("api_key", "")
        max_papers = int(data.get("max_papers", 6))
        max_papers = max(3, min(max_papers, 8))  # keep LLM calls bounded
        if not topic:
            return jsonify({"error": "Missing topic"}), 400

        generator = LearningPathGenerator(_TfidfEmbedder(), _LLMRouter(api_key), _AbstractRAG())
        # Supply a 'year' field derived from the published date for difficulty heuristics
        enriched = []
        for p in papers_db:
            q = dict(p)
            pub = str(p.get("published", ""))
            q["year"] = int(pub[:4]) if pub[:4].isdigit() else 2024
            q["summary"] = p.get("abstract", "")[:200]
            enriched.append(q)

        result = generator.generate_path(topic, enriched, max_papers=max_papers)
        append_reward_event("learning_path_generated", topic, result["metadata"]["num_papers"])
        return jsonify(result)
    except Exception as e:
        print(f"Learning path error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/notes", methods=["GET", "POST"])
def api_notes():
    """Semantic research notebook (create / update / delete / search / list)."""
    try:
        mgr = NotebookManager(_TfidfEmbedder(), _LLMRouter(""), storage_path=NOTEBOOK_FILE)
        # Rebuild the paper index from loaded notes
        for nid, note in mgr.notes.items():
            mgr.paper_index.setdefault(note.paper_id, []).append(nid)

        if request.method == "GET":
            paper_id = request.args.get("paper_id", "")
            query = request.args.get("query", "")
            if query:
                return jsonify({"notes": mgr.search_notes(query)})
            if paper_id:
                return jsonify({"notes": mgr.get_notes_for_paper(paper_id)})
            all_notes = [_dc_asdict(n) for n in mgr.notes.values()]
            all_notes.sort(key=lambda n: n["updated_at"], reverse=True)
            return jsonify({"notes": all_notes})

        data = request.get_json() or {}
        action = data.get("action", "create")
        if action == "create":
            note = mgr.create_note(
                data.get("paper_id", ""),
                data.get("title", "Untitled note"),
                data.get("content", ""),
                tags=data.get("tags", []),
            )
            return jsonify({"status": "success", "note": _dc_asdict(note)})
        elif action == "update":
            note = mgr.update_note(
                data.get("note_id", ""),
                title=data.get("title"),
                content=data.get("content"),
                tags=data.get("tags"),
            )
            return jsonify({"status": "success", "note": _dc_asdict(note)})
        elif action == "delete":
            ok = mgr.delete_note(data.get("note_id", ""))
            return jsonify({"status": "success" if ok else "not_found"})
        elif action == "search":
            return jsonify({"notes": mgr.search_notes(data.get("query", ""))})
        return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        print(f"Notes error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/notes/related/<note_id>", methods=["GET"])
def api_notes_related(note_id):
    """Papers related to a note via semantic similarity of its embedding."""
    try:
        embedder = _SemanticEmbedder()
        mgr = NotebookManager(embedder, _LLMRouter(""), storage_path=NOTEBOOK_FILE)
        for nid, n in mgr.notes.items():
            mgr.paper_index.setdefault(n.paper_id, []).append(nid)
        note = mgr.notes.get(note_id)
        if not note:
            return jsonify({"error": "Note not found"}), 404

        # Use the stored semantic embedding of the note
        q_emb = np.array(note.embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scored = []
        for p in papers_db:
            if p.get("id") == note.paper_id:
                continue
            # Encode each paper's title + abstract semantically
            paper_text = (p.get("title", "") + " " + p.get("abstract", "")).strip()
            p_emb = embedder.encode(paper_text)
            p_norm = np.linalg.norm(p_emb)
            if p_norm > 0:
                p_emb = p_emb / p_norm
            sim = float(np.dot(q_emb, p_emb))
            scored.append((sim, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = [{"id": p["id"], "title": p["title"], "score": round(s, 3),
                "category_name": p.get("category_name", "")} for s, p in scored[:5]]
        return jsonify({"papers": out})
    except Exception as e:
        print(f"Related notes error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/content", methods=["POST"])
def api_content():
    """Paper-to-Content Studio: ELI5 / slides / tweets / podcast / blog / infographic."""
    try:
        data = request.get_json() or {}
        paper_id = data.get("paper_id", "")
        fmt = data.get("format", "eli5")
        api_key = data.get("api_key", "")
        audience = data.get("audience")
        paper = next((p for p in papers_db if p.get("id") == paper_id), None)
        if not paper:
            return jsonify({"error": "Paper not found"}), 404
        try:
            fmt_enum = ContentFormat(fmt)
        except ValueError:
            return jsonify({"error": f"Unknown format '{fmt}'"}), 400

        studio = PaperContentStudio(_LLMRouter(api_key), _AbstractRAG())
        output = studio.generate_format(paper_id, paper, fmt_enum, audience)
        append_reward_event("content_generated", paper_id, fmt)
        return jsonify(_dc_asdict(output))
    except Exception as e:
        print(f"Content studio error: {e}")
        return jsonify({"error": str(e)}), 500


def _papers_for_network():
    """Normalize papers into the shape AuthorVenueNetwork expects
    (authors as dicts, venue as a dict keyed on the arXiv category)."""
    out = []
    for p in papers_db:
        authors = p.get("authors", [])
        norm_authors = [{"name": a} if isinstance(a, str) else a for a in authors]
        cat = p.get("category_name") or p.get("primary_category") or "arXiv"
        out.append({
            **p,
            "authors": norm_authors,
            "venue": {"id": (p.get("primary_category") or cat).lower().replace(" ", "_"),
                      "name": cat, "type": "arXiv category", "impact_factor": 0.0},
            "citation_count": p.get("citation_count", 0),
        })
    return out


@app.route("/api/network", methods=["GET"])
def api_network():
    """Author & venue co-authorship / topic network for force-directed graph."""
    try:
        top_authors = int(request.args.get("top_authors", 25))
        min_collab = int(request.args.get("min_collaboration", 1))
        net = AuthorVenueNetwork(_papers_for_network())
        graph = net.get_network_graph(top_authors=top_authors, min_collaboration=min_collab)
        return jsonify(graph)
    except Exception as e:
        print(f"Network error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/network/author/<author_id>", methods=["GET"])
def api_network_author(author_id):
    try:
        net = AuthorVenueNetwork(_papers_for_network())
        profile = net.get_author_profile(author_id)
        if profile is None:
            return jsonify({"error": "Author not found"}), 404
        return jsonify(profile)
    except Exception as e:
        print(f"Author profile error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlists", methods=["GET", "POST"])
def api_watchlists():
    """Watchlists & alerts: follow keywords/authors/venues, scan for new matches."""
    try:
        mgr = WatchlistManager(_TfidfEmbedder(), _LLMRouter(""), storage_path=WATCHLIST_FILE)

        if request.method == "GET":
            watchlists = [_dc_asdict(w) for w in mgr.watchlists.values()]
            return jsonify({
                "watchlists": watchlists,
                "unread_alerts": mgr.get_unread_alerts(),
                "summary": mgr.get_alert_summary(),
            })

        data = request.get_json() or {}
        action = data.get("action", "")
        if action == "create":
            wl = mgr.create_watchlist(data.get("name", "My Watchlist"), data.get("description", ""))
            return jsonify({"status": "success", "watchlist": _dc_asdict(wl)})
        elif action == "add_item":
            ok = mgr.add_watch_item(data.get("watchlist_id", ""), data.get("type", "keyword"), data.get("value", ""))
            return jsonify({"status": "success" if ok else "exists_or_missing"})
        elif action == "remove_item":
            ok = mgr.remove_watch_item(data.get("watchlist_id", ""), data.get("type", "keyword"), data.get("value", ""))
            return jsonify({"status": "success" if ok else "not_found"})
        elif action == "scan":
            # Scan the newest papers for matches against all watchlists
            recent = sorted(papers_db, key=lambda p: str(p.get("published", "")), reverse=True)[:150]
            alerts = mgr.process_new_papers(recent, data.get("watchlist_ids"))
            return jsonify({"status": "success", "new_alerts": [_dc_asdict(a) for a in alerts], "count": len(alerts)})
        elif action == "mark_read":
            ok = mgr.mark_alert_read(data.get("alert_id", ""))
            return jsonify({"status": "success" if ok else "not_found"})
        elif action == "mark_all_read":
            count = mgr.mark_all_read(data.get("watchlist_id"))
            return jsonify({"status": "success", "marked": count})
        return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        print(f"Watchlist error: {e}")
        return jsonify({"error": str(e)}), 500


import subprocess, shutil

_SPD_SAY = shutil.which("spd-say")

@app.route("/api/tts/status", methods=["GET"])
def api_tts_status():
    """Report whether server-side text-to-speech (speech-dispatcher) is available.
    Used by the Audio Digest as a fallback when the browser exposes no voices
    (a common Chromium-on-Linux limitation)."""
    return jsonify({"available": bool(_SPD_SAY)})


@app.route("/api/tts/speak", methods=["POST"])
def api_tts_speak():
    """Speak text through the host machine's speakers via spd-say (non-blocking).
    Intended for local use: the browser and backend run on the same machine."""
    if not _SPD_SAY:
        return jsonify({"error": "No server-side TTS engine (spd-say) installed"}), 501
    try:
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Missing text"}), 400
        text = text[:20000]  # keep the utterance queue sane
        try:
            rate = float(data.get("rate", 1.0))
        except (TypeError, ValueError):
            rate = 1.0
        spd_rate = max(-100, min(100, int((rate - 1.0) * 100)))

        # Cancel anything currently playing, then queue the new text.
        subprocess.run([_SPD_SAY, "-C"], timeout=5)
        # List-form args (no shell); '--' stops option parsing so text can't look like a flag.
        cmd = [_SPD_SAY, "-r", str(spd_rate), "-l", data.get("lang", "en"), "--", text]
        subprocess.Popen(cmd)  # fire-and-forget; spd-say hands off to the daemon
        return jsonify({"status": "speaking"})
    except Exception as e:
        print(f"TTS speak error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts/stop", methods=["POST"])
def api_tts_stop():
    """Stop/cancel any server-side speech."""
    if not _SPD_SAY:
        return jsonify({"status": "noop"})
    try:
        subprocess.run([_SPD_SAY, "-C"], timeout=5)
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Report which recommender features are active + semantic coverage."""
    covered = sum(1 for p in papers_db if p["id"] in semantic_vectors)
    return jsonify({
        "semantic_active": USE_SEMANTIC,
        "semantic_covered": covered,
        "total_papers": len(papers_db),
        "ranker": "semantic+LR" if USE_SEMANTIC else "tfidf+LR",
    })


@app.route("/api/build-semantic", methods=["POST"])
def api_build_semantic():
    """Compute & cache Gemini semantic embeddings for the corpus, then retrain
    the recommender to use them. Safe to call repeatedly (resumes where it left off)."""
    global papers_db
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "A Gemini API key is required to build semantic embeddings."}), 400
    try:
        result = build_semantic_cache(papers_db, api_key)
        _load_semantic_state()
        if ratings_db:
            papers_db = train_recommender(papers_db, ratings_db, vectors=_rec_vectors())
            save_papers_to_disk()
        result["semantic_active"] = USE_SEMANTIC
        if result["covered"] < result["total"]:
            result["note"] = ("Partial — the embedding API returned fewer vectors than requested "
                              "(likely rate-limited). Call again to resume.")
        return jsonify(result)
    except Exception as e:
        print(f"Build-semantic error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh-papers", methods=["POST"])
def api_refresh_papers():
    """Re-crawl the newest papers from arXiv and rebuild TF-IDF / t-SNE / recommender.
    (The corpus is otherwise a cached snapshot; this pulls fresh, recent papers.)"""
    # Back up the current (good) corpus so a failed/rate-limited crawl can't destroy it.
    backup = None
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "rb") as f:
                backup = f.read()
    except Exception:
        backup = None

    def _restore_and_reload():
        if backup is not None:
            try:
                with open(DATA_FILE, "wb") as f:
                    f.write(backup)
            except Exception:
                pass
        init_db()

    try:
        # Force a re-crawl by removing the cache, then rebuild everything.
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        init_db()

        # Guard: arXiv rate-limits (HTTP 429) and the crawler then returns nothing,
        # which would otherwise leave only the tiny built-in fallback list. If we
        # didn't get a healthy corpus, roll back to the previous data.
        if len(papers_db) < 100:
            _restore_and_reload()
            return jsonify({
                "error": "arXiv is rate-limiting or unavailable right now, so the refresh was cancelled. "
                         f"Your existing {len(papers_db)} papers were kept — please try again in a few minutes.",
                "count": len(papers_db),
            }), 503

        dates = sorted([p.get("published", "")[:10] for p in papers_db if p.get("published")])
        append_reward_event("papers_refreshed", "all", len(papers_db))
        return jsonify({
            "status": "success",
            "count": len(papers_db),
            "newest": dates[-1] if dates else None,
            "oldest": dates[0] if dates else None,
        })
    except Exception as e:
        print(f"Refresh error: {e}")
        _restore_and_reload()
        return jsonify({"error": f"Refresh failed ({e}). Your existing papers were kept."}), 500


@app.route("/api/data-freshness", methods=["GET"])
def api_data_freshness():
    """Report the date range of the currently loaded corpus."""
    dates = sorted([p.get("published", "")[:10] for p in papers_db if p.get("published")])
    return jsonify({
        "count": len(papers_db),
        "newest": dates[-1] if dates else None,
        "oldest": dates[0] if dates else None,
    })


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
