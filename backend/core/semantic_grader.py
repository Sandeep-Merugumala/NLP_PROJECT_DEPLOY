"""
Semantic Textual Similarity (STS) Grader
=========================================
Replaces manual self-grading (0-4) with automatic grading using
Sentence-BERT (SBERT) cosine similarity.

NLP Concept: Semantic Textual Similarity
Model: all-MiniLM-L6-v2 (fast, 384-dim sentence embeddings)

The user types their answer in a text box.  This module computes the
cosine similarity between the student's typed answer and the reference
answer, then maps that similarity score to the SM-2 quality scale (0-4).
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton so the 80 MB model is downloaded only once
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Lazy-load the Sentence-BERT model (downloaded on first call)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-BERT model loaded successfully.")
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two pieces of text using SBERT.

    Returns a float in [0, 1] where 1 means identical meaning.
    """
    model = _get_model()

    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)

    # Cosine similarity
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
    )

    # Clamp to [0, 1] (negative similarity treated as 0)
    return float(max(0.0, min(1.0, cos_sim)))


def similarity_to_quality(similarity: float) -> int:
    """
    Map cosine similarity ∈ [0, 1] to the SM-2 quality scale (0-4).

    Thresholds (tuned for academic Q&A):
        similarity >= 0.75  →  4  (Excellent / trivially easy)
        similarity >= 0.55  →  3  (Good / correct with effort)
        similarity >= 0.40  →  2  (Acceptable / partially correct)
        similarity >= 0.25  →  1  (Difficult / mostly wrong)
        similarity <  0.25  →  0  (Complete failure / blank)
    """
    if similarity >= 0.75:
        return 4
    elif similarity >= 0.55:
        return 3
    elif similarity >= 0.40:
        return 2
    elif similarity >= 0.25:
        return 1
    else:
        return 0


def grade_answer(
    user_answer: str,
    reference_answer: str,
) -> dict:
    """
    Grade a user's typed answer against the reference answer.

    Returns:
        {
            "similarity": float,        # raw cosine similarity (0-1)
            "quality": int,             # SM-2 quality score (0-4)
            "correct": bool,            # True if quality >= 3
            "feedback": str,            # human-readable feedback
            "method": "sbert_sts"       # grading method identifier
        }
    """
    # Edge case: empty or very short answer
    if not user_answer or len(user_answer.strip()) < 3:
        return {
            "similarity": 0.0,
            "quality": 0,
            "correct": False,
            "feedback": "💭 No answer provided. Try to recall the key concepts!",
            "method": "sbert_sts",
        }

    similarity = compute_similarity(user_answer.strip(), reference_answer.strip())
    quality = similarity_to_quality(similarity)
    correct = quality >= 3

    # Generate feedback based on quality
    feedback_map = {
        4: f"🎉 Excellent! Your answer is highly similar to the reference (similarity: {similarity:.0%}).",
        3: f"✅ Good job! You captured the core idea (similarity: {similarity:.0%}).",
        2: f"⚠️ Partially correct. Some key concepts are missing (similarity: {similarity:.0%}).",
        1: f"❌ Your answer diverges from the expected one (similarity: {similarity:.0%}). Review this topic.",
        0: f"💪 Not quite right (similarity: {similarity:.0%}). Let's focus on this concept.",
    }

    return {
        "similarity": round(similarity, 4),
        "quality": quality,
        "correct": correct,
        "feedback": feedback_map[quality],
        "method": "sbert_sts",
    }
