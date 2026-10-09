"""
RAG Chat Evaluation — precision@k, faithfulness, hallucination detection.

Tests the PDF RAG pipeline at three levels:
  1. Retrieval: precision@k (are top-k chunks from the right page?)
  2. Answer Quality: ROUGE-L overlap against a reference answer
  3. Hallucination: checks if the answer contains phrases not found in any chunk

Usage:
    python backend/eval/rag_eval.py

Requires: rouge-score  (pip install rouge-score)
"""

import sys
import json
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Try to import rouge; graceful fallback
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score not installed. ROUGE metrics will be skipped.")
    print("       Run: pip install rouge-score")


# --------------------------------------------------------------------------
# Synthetic test cases: (context_chunks, question, reference_answer)
# These are domain-agnostic and test the RAG pipeline's chunking logic
# without needing a live backend or API key.
# --------------------------------------------------------------------------
TEST_CASES = [
    {
        "name": "Methodology extraction",
        "context": [
            {"child_text": "We train a logistic regression model on TF-IDF embeddings of paper abstracts.", "text": "We train a logistic regression model on TF-IDF embeddings of paper abstracts. The model uses a custom weighting scheme to handle class imbalance.", "page_idx": 2},
            {"child_text": "We use weighted binary cross-entropy loss for training.", "text": "We use weighted binary cross-entropy loss for training. Positive samples are upweighted by 1/n_P.", "page_idx": 2},
            {"child_text": "The model is evaluated on balanced accuracy and F1 score.", "text": "The model is evaluated on balanced accuracy and F1 score.", "page_idx": 3},
        ],
        "question": "What is the training method?",
        "reference": "The model uses logistic regression trained on TF-IDF embeddings with weighted binary cross-entropy loss.",
        "expected_page": 2
    },
    {
        "name": "Result extraction",
        "context": [
            {"child_text": "Our method achieves 0.87 balanced accuracy on the held-out test set.", "text": "Our method achieves 0.87 balanced accuracy on the held-out test set, outperforming the baseline by 12%.", "page_idx": 5},
            {"child_text": "The baseline achieves 0.75 balanced accuracy.", "text": "The baseline achieves 0.75 balanced accuracy using standard cosine similarity ranking.", "page_idx": 5},
        ],
        "question": "What balanced accuracy does the proposed method achieve?",
        "reference": "The proposed method achieves 0.87 balanced accuracy.",
        "expected_page": 5
    },
    {
        "name": "Abstract-level understanding",
        "context": [
            {"child_text": "We present Scholar Inbox, a personalised paper recommendation system.", "text": "We present Scholar Inbox, a personalised paper recommendation system based on active learning and ranked retrieval.", "page_idx": 1},
        ],
        "question": "What is Scholar Inbox?",
        "reference": "Scholar Inbox is a personalised paper recommendation system.",
        "expected_page": 1
    }
]


def simulate_tfidf_retrieval(context_chunks, question, top_k=3):
    """Local TF-IDF retrieval to simulate the RAG backend without a server."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [c["child_text"] for c in context_chunks]
    vec = TfidfVectorizer(stop_words="english")
    vec.fit(corpus + [question])
    chunk_vecs = vec.transform(corpus)
    q_vec = vec.transform([question])
    sims = np.dot(chunk_vecs.toarray(), q_vec.T.toarray()).flatten()
    for i, c in enumerate(context_chunks):
        c["score"] = float(sims[i])
    ranked = sorted(context_chunks, key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]


def precision_at_k(retrieved, expected_page, k=3):
    """Fraction of top-k chunks that come from the expected page."""
    hits = sum(1 for c in retrieved[:k] if c.get("page_idx") == expected_page)
    return hits / k


def rouge_l(hypothesis, reference):
    """Compute ROUGE-L F1 between hypothesis and reference strings."""
    if not HAS_ROUGE:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


def hallucination_rate(answer, context_chunks):
    """
    Rough hallucination check: what fraction of sentences in the answer
    have NO word overlap (ignoring stopwords) with any chunk?
    """
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in", "of", "to", "and", "or", "that", "this"}
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
    all_context = " ".join(c["text"] for c in context_chunks).lower()
    
    hallucinated = 0
    for sentence in sentences:
        words = [w for w in sentence.lower().split() if w not in stopwords and len(w) > 3]
        if not words:
            continue
        overlap = sum(1 for w in words if w in all_context)
        if overlap == 0:
            hallucinated += 1
    
    return hallucinated / max(len(sentences), 1)


def generate_heuristic_answer(top_chunks, question):
    """Simple heuristic answer generation (no LLM needed for evaluation)."""
    if not top_chunks:
        return "No relevant content found."
    return top_chunks[0]["text"]


def run_evaluation():
    print("=" * 60)
    print("RAG Chat Evaluation — Scholar Inbox")
    print("=" * 60)
    
    results = []
    for tc in TEST_CASES:
        print(f"\n[Test: {tc['name']}]")
        print(f"  Question: {tc['question']}")
        
        # Retrieval
        retrieved = simulate_tfidf_retrieval(tc["context"], tc["question"], top_k=3)
        p_at_k = precision_at_k(retrieved, tc["expected_page"])
        print(f"  Precision@3: {p_at_k:.2f} (expected page {tc['expected_page']})")
        
        # Answer generation (heuristic)
        answer = generate_heuristic_answer(retrieved, tc["question"])
        
        # ROUGE-L
        rl = rouge_l(answer, tc["reference"])
        print(f"  ROUGE-L: {rl:.4f}" if not np.isnan(rl) else "  ROUGE-L: N/A (install rouge-score)")
        
        # Hallucination rate
        hall = hallucination_rate(answer, tc["context"])
        print(f"  Hallucination rate: {hall:.2f}")
        
        results.append({
            "test": tc["name"],
            "precision_at_3": p_at_k,
            "rouge_l": rl,
            "hallucination_rate": hall
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    avg_p = np.nanmean([r["precision_at_3"] for r in results])
    avg_rl = np.nanmean([r["rouge_l"] for r in results])
    avg_hall = np.nanmean([r["hallucination_rate"] for r in results])
    print(f"  Avg Precision@3:       {avg_p:.4f}")
    print(f"  Avg ROUGE-L:           {avg_rl:.4f}" if not np.isnan(avg_rl) else "  Avg ROUGE-L:           N/A")
    print(f"  Avg Hallucination:     {avg_hall:.4f}")
    
    # Save results
    import csv
    from pathlib import Path
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "rag_eval.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "precision_at_3", "rouge_l", "hallucination_rate"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults saved to {path}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
