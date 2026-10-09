"""
NLP Question Generation Pipeline
==================================
Replaces generic LLM prompting with a specialized two-stage NLP pipeline:

Stage 1 — Keyphrase Extraction (KeyBERT)
    Identifies the most important technical terms / concepts in the paper
    using BERT-based keyword extraction.

Stage 2 — Question Generation (T5)
    Uses a T5 model fine-tuned for question generation to create specific,
    targeted questions from the paper text and extracted keyphrases.

NLP Concepts: Keyphrase Extraction, Conditional Text Generation, Seq2Seq
"""

import logging
import re
import random
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_keybert_model = None
_qg_pipeline = None


def _get_keybert():
    """Lazy-load KeyBERT model."""
    global _keybert_model
    if _keybert_model is None:
        try:
            from keybert import KeyBERT
            logger.info("Loading KeyBERT model...")
            _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
            logger.info("KeyBERT model loaded successfully.")
        except ImportError:
            logger.error("keybert is not installed. Run: pip install keybert")
            raise
    return _keybert_model


def _get_qg_pipeline():
    """Lazy-load T5 question generation model and tokenizer."""
    global _qg_pipeline
    if _qg_pipeline is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("Loading T5 question generation model...")
            model_name = "valhalla/t5-small-qg-prepend"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _qg_pipeline = (model, tokenizer)
            logger.info("T5 QG model loaded successfully.")
        except ImportError:
            logger.error(
                "transformers is not installed. Run: pip install transformers"
            )
            raise
    return _qg_pipeline


# ---------------------------------------------------------------------------
# Stage 1: Keyphrase Extraction
# ---------------------------------------------------------------------------

def extract_keyphrases(
    text: str,
    top_n: int = 10,
    keyphrase_ngram_range: tuple = (1, 3),
    diversity: float = 0.5,
) -> List[Dict]:
    """
    Extract keyphrases from text using KeyBERT.

    Returns a list of dicts:
        [{"keyphrase": "...", "score": 0.85}, ...]
    """
    kw_model = _get_keybert()

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=top_n,
        use_mmr=True,           # Maximal Marginal Relevance for diversity
        diversity=diversity,
    )

    return [{"keyphrase": kw, "score": round(score, 4)} for kw, score in keywords]


# ---------------------------------------------------------------------------
# Stage 2: T5 Question Generation
# ---------------------------------------------------------------------------

def _generate_question_t5(context: str, answer: str) -> str:
    """
    Generate a question from context + answer using the T5 QG model.

    The model expects input in the format:
        "answer: <answer> context: <context>"
    """
    model, tokenizer = _get_qg_pipeline()

    # T5 QG prepend format
    input_text = f"answer: {answer} context: {context}"

    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=128, num_beams=4)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Clean up: remove "question: " prefix if present
        question = re.sub(r"^question:\s*", "", question, flags=re.IGNORECASE)
        return question
    except Exception as e:
        logger.warning(f"T5 QG failed: {e}")
        return f"What is {answer}?"


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


# ---------------------------------------------------------------------------
# Main Pipeline: KeyBERT + T5 Question Generation
# ---------------------------------------------------------------------------

def generate_nlp_cards(
    paper_id: str,
    title: str,
    abstract: str,
    num_cards: int = 10,
) -> Dict:
    """
    Generate study cards using NLP pipeline (KeyBERT + T5).

    Pipeline:
        1. Extract keyphrases from the abstract using KeyBERT
        2. For each keyphrase, use T5 to generate a question
        3. Build MCQ distractors from other keyphrases
        4. Create flashcard Q&A pairs

    Returns:
        {"questions": [...], "flashcards": [...]}
    """

    full_text = f"{title}. {abstract}"
    sentences = _split_sentences(abstract)

    # --- Stage 1: KeyBERT Keyphrase Extraction ---
    keyphrases = extract_keyphrases(full_text, top_n=15)
    logger.info(f"Extracted {len(keyphrases)} keyphrases: {[kp['keyphrase'] for kp in keyphrases[:5]]}")

    if not keyphrases:
        # Fallback: use title words as keyphrases
        words = [w for w in title.split() if len(w) > 4]
        keyphrases = [{"keyphrase": w, "score": 0.5} for w in words[:5]]

    # --- Stage 2: Generate Questions ---
    questions = []
    flashcards = []

    # Pick the best sentences as context for question generation
    # Use sentences that contain keyphrases for better context
    keyphrase_texts = [kp["keyphrase"] for kp in keyphrases]

    # Generate MCQs (7 cards)
    mcq_count = min(7, len(keyphrases))
    used_keyphrases = set()

    for i in range(mcq_count):
        kp = keyphrases[i]
        answer_text = kp["keyphrase"]

        if answer_text.lower() in used_keyphrases:
            continue
        used_keyphrases.add(answer_text.lower())

        # Find the best sentence containing this keyphrase
        context = abstract[:500]  # default context
        for sent in sentences:
            if answer_text.lower() in sent.lower():
                context = sent
                break

        # Generate question using T5
        question = _generate_question_t5(context, answer_text)

        # Build MCQ distractors from other keyphrases
        distractors = [
            kp2["keyphrase"]
            for kp2 in keyphrases
            if kp2["keyphrase"].lower() != answer_text.lower()
        ][:3]

        # Pad distractors if needed
        fallback_distractors = [
            "baseline model", "random sampling", "gradient descent",
            "cross-entropy loss", "attention mechanism", "fine-tuning",
            "data augmentation", "transfer learning"
        ]
        while len(distractors) < 3:
            for fd in fallback_distractors:
                if fd not in distractors and fd.lower() != answer_text.lower():
                    distractors.append(fd)
                    break
            else:
                break

        # Shuffle options
        options = [answer_text] + distractors[:3]
        random.shuffle(options)

        # Find the correct answer label
        labels = ["A", "B", "C", "D"]
        correct_label = labels[options.index(answer_text)]

        questions.append({
            "q": question,
            "options": [f"{labels[j]}) {options[j]}" for j in range(len(options))],
            "answer": correct_label,
            "keyphrase": answer_text,
            "keyphrase_score": kp["score"],
        })

    # Generate Flashcards (3 cards)
    flashcard_keyphrases = keyphrases[mcq_count: mcq_count + 3]
    if len(flashcard_keyphrases) < 3:
        flashcard_keyphrases = keyphrases[:3]

    for kp in flashcard_keyphrases:
        answer_text = kp["keyphrase"]

        # Find sentence containing the keyphrase as the answer
        answer_sentence = None
        for sent in sentences:
            if answer_text.lower() in sent.lower():
                answer_sentence = sent
                break
        if not answer_sentence:
            answer_sentence = f"{answer_text} is a key concept discussed in '{title}'."

        # Generate question using T5
        question = _generate_question_t5(abstract[:500], answer_text)

        flashcards.append({
            "q": question,
            "a": answer_sentence,
            "keyphrase": answer_text,
            "keyphrase_score": kp["score"],
        })

    # Ensure we have at least some cards
    if not questions and not flashcards:
        flashcards.append({
            "q": f"What is the main contribution of '{title}'?",
            "a": abstract[:240] if abstract else title,
        })

    logger.info(
        f"NLP pipeline generated {len(questions)} MCQs + {len(flashcards)} flashcards "
        f"for paper {paper_id}"
    )

    return {"questions": questions, "flashcards": flashcards}
