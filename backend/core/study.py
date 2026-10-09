"""
Study Mode
Generate MCQs and flashcards from papers with SM-2 spaced repetition scheduling
and Thompson Sampling RL for adaptive card selection.

NLP Pipeline (KeyBERT + T5) is used as the primary card generation method.
Semantic Grader (SBERT STS) provides automatic answer grading.
Falls back to LLM prompting if NLP models are unavailable.
"""

import json
import math
import random
import os
import logging
from datetime import datetime, timedelta
from .llm_helper import call_llm_with_retry

logger = logging.getLogger(__name__)

STUDY_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "study.jsonl")

class Card:
    """Represents a study card (question + answer)."""
    
    def __init__(self, paper_id, question, answer, card_type='mcq', options=None):
        self.id = f"{paper_id}_{hash(question) % 10000}"
        self.paper_id = paper_id
        self.question = question
        self.answer = answer
        self.card_type = card_type  # 'mcq' or 'flashcard'
        self.options = options or []

        
        # SM-2 parameters
        self.ease_factor = 2.5
        self.interval = 1  # days
        self.repetitions = 0
        self.last_reviewed = None
        self.next_review = datetime.now()
        
        # RL parameters
        self.alpha = 1.0  # Thompson Sampling parameter (successes + 1)
        self.beta = 1.0   # Thompson Sampling parameter (failures + 1)
        self.correct_count = 0
        self.incorrect_count = 0
        self.topic = ""
    
    def to_dict(self):
        """Serialize card to dict."""
        return {
            'id': self.id,
            'paper_id': self.paper_id,
            'question': self.question,
            'answer': self.answer,
            'card_type': self.card_type,
            'options': self.options,
            'ease_factor': self.ease_factor,
            'interval': self.interval,
            'repetitions': self.repetitions,
            'last_reviewed': self.last_reviewed.isoformat() if self.last_reviewed else None,
            'next_review': self.next_review.isoformat(),
            'alpha': self.alpha,
            'beta': self.beta,
            'correct_count': self.correct_count,
            'incorrect_count': self.incorrect_count,
            'topic': self.topic
        }
    
    @staticmethod
    def from_dict(data):
        """Deserialize card from dict."""
        card = Card(data['paper_id'], data['question'], data['answer'], data.get('card_type', 'mcq'), data.get('options', []))
        card.id = data['id']
        card.ease_factor = data.get('ease_factor', 2.5)
        card.interval = data.get('interval', 1)
        card.repetitions = data.get('repetitions', 0)
        
        if data.get('last_reviewed'):
            card.last_reviewed = datetime.fromisoformat(data['last_reviewed'])
        
        card.next_review = datetime.fromisoformat(data.get('next_review', datetime.now().isoformat()))
        card.alpha = data.get('alpha', 1.0)
        card.beta = data.get('beta', 1.0)
        card.correct_count = data.get('correct_count', 0)
        card.incorrect_count = data.get('incorrect_count', 0)
        card.topic = data.get('topic', '')
        
        return card


class SM2Scheduler:
    """Implements SM-2 spaced repetition algorithm."""
    
    @staticmethod
    def update_card(card, quality, days_since_last=1):
        """
        Update card based on quality of recall.
        quality: 0-5 (0-1 = fail, 2-4 = hard/ok/easy, 5 = perfect)
        """
        if quality < 2:
            # Forgot - reset
            card.interval = 1
            card.repetitions = 0
            card.ease_factor = max(1.3, card.ease_factor - 0.2)
        else:
            card.repetitions += 1
            if card.repetitions == 1:
                card.interval = 1
            elif card.repetitions == 2:
                card.interval = 3
            else:
                card.interval = int(card.interval * card.ease_factor)
            
            # Ease factor adjustment
            ease_delta = 0.1 * (5 - quality)
            card.ease_factor = max(1.3, card.ease_factor + ease_delta)
        
        card.last_reviewed = datetime.now()
        card.next_review = datetime.now() + timedelta(days=card.interval)
    
    @staticmethod
    def get_due_cards(cards):
        """Get cards due for review today."""
        now = datetime.now()
        return [c for c in cards if c.next_review <= now]


class RLCardScheduler:
    """Thompson Sampling bandit for adaptive card selection."""
    
    @staticmethod
    def sample_arm(card):
        """Sample success probability from Beta distribution."""
        return random.betavariate(card.alpha, card.beta)
    
    @staticmethod
    def select_next_card(due_cards, topic_mastery=None):
        """
        Select next card using Thompson Sampling.
        Prioritize cards at learning frontier (not too easy, not too hard).
        
        Args:
            due_cards: List of Card objects
            topic_mastery: Optional dict of topic -> mastery_score (0-1)
        
        Returns:
            Selected Card or None
        """
        if not due_cards:
            return None
        
        if len(due_cards) == 1:
            return due_cards[0]
        
        # Sample success probability for each card
        sampled_values = []
        for card in due_cards:
            # Base Thompson Sampling
            ts_sample = RLCardScheduler.sample_arm(card)
            
            # Adjust for learning frontier heuristic
            # Prefer cards that are at ~50% success rate (hardest)
            frontier_bonus = 1.0 - abs(0.5 - ts_sample)  # Closer to 50% = higher score
            
            adjusted_value = ts_sample * 0.7 + frontier_bonus * 0.3
            sampled_values.append(adjusted_value)
        
        # Select card with highest adjusted value
        max_idx = sampled_values.index(max(sampled_values))
        return due_cards[max_idx]
    
    @staticmethod
    def update_card_rl(card, correct, reward=None):
        """
        Update RL parameters (Thompson Sampling) after card review.
        
        Args:
            card: Card object
            correct: Boolean indicating if answer was correct
            reward: Optional custom reward (default: 1 if correct at frontier, else 0)
        """
        if reward is None:
            # Default: +1 reward if correct at learning frontier
            success_rate = card.correct_count / max(1, card.correct_count + card.incorrect_count)
            frontier_score = 1.0 - abs(0.5 - success_rate)
            reward = 1.0 if correct else 0.0
            reward *= frontier_score
        
        if correct:
            card.alpha += reward
            card.correct_count += 1
        else:
            card.beta += (1 - reward)
            card.incorrect_count += 1


def generate_cards_from_paper(paper, abstract_only=False, api_key=None, use_groq=False):
    """
    Generate study cards (MCQs and flashcards) from a paper.
    
    Uses a two-stage NLP pipeline as the PRIMARY method:
        Stage 1: KeyBERT keyphrase extraction
        Stage 2: T5 question generation
    
    Falls back to LLM prompting if NLP models fail, and finally
    to rule-based cloze cards if everything else fails.
    
    Args:
        paper: Paper dict with 'id', 'title', 'abstract'
        abstract_only: If True, use only abstract
        api_key: Optional API key for LLM fallback
        use_groq: Use Groq instead of Gemini for LLM fallback
    
    Returns:
        List of Card objects
    """
    paper_id = paper.get('id', '')
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')

    # =====================================================================
    # PRIMARY: NLP Pipeline (KeyBERT + T5 Question Generation)
    # =====================================================================
    try:
        from .nlp_question_gen import generate_nlp_cards
        logger.info(f"Using NLP pipeline (KeyBERT + T5) for paper: {paper_id}")
        data = generate_nlp_cards(paper_id, title, abstract, num_cards=10)

        cards = []
        for mcq in data.get('questions', []):
            raw_options = mcq.get('options', [])
            if isinstance(raw_options, str):
                if '\n' in raw_options:
                    raw_options = [opt.strip() for opt in raw_options.split('\n') if opt.strip()]
                elif raw_options.count(',') >= 3:
                    raw_options = [opt.strip() for opt in raw_options.split(',') if opt.strip()]
                else:
                    raw_options = [raw_options]

            card = Card(
                paper_id,
                mcq['q'],
                mcq.get('answer', ''),
                card_type='mcq',
                options=raw_options
            )
            card.topic = extract_topic(title)
            cards.append(card)

        for fc in data.get('flashcards', []):
            card = Card(
                paper_id,
                fc['q'],
                fc['a'],
                card_type='flashcard'
            )
            card.topic = extract_topic(title)
            cards.append(card)

        if len(cards) >= 3:
            # Pad to 10 if needed
            if len(cards) < 10:
                cards.extend(_cards_from_dict(paper_id, title, _fallback_cards(title, abstract)))
            logger.info(f"NLP pipeline generated {len(cards[:10])} cards successfully.")
            return cards[:10]
        else:
            logger.warning("NLP pipeline produced too few cards, trying LLM fallback.")

    except Exception as e:
        logger.warning(f"NLP pipeline failed ({e}), falling back to LLM.")

    # =====================================================================
    # FALLBACK 1: LLM Prompting (Gemini / Groq)
    # =====================================================================
    content = abstract if abstract_only else f"TITLE: {title}\n\nABSTRACT: {abstract}"
    
    prompt = f"""You are an expert tutor. Generate exactly 10 high-quality study cards for this research paper:

{content}

Create a mix of:
1. Seven (7) multiple choice questions. Each must have 4 distinct, plausible options labelled "A) ...", "B) ...", "C) ...", "D) ..." and a single-letter answer key (A/B/C/D). Cover the problem, method, datasets, results, and limitations. Make the distractors non-trivial.
2. Three (3) flashcards (question | answer pair) testing the core concepts, key terminology, and the main takeaway.

Return ONLY valid JSON, no markdown fences:
{{"questions": [{{"q": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A"}}], "flashcards": [{{"q": "...", "a": "..."}}]}}"""

    try:
        if api_key:
            response = call_llm_with_retry(prompt, api_key, use_groq=use_groq)
        else:
            response = None

        if response:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(json_match.group() if json_match else response)
        else:
            # Fallback data (no/failed LLM): build up to 10 cards from the abstract itself
            data = _fallback_cards(title, abstract)

        cards = []
        
        # Add MCQs
        for mcq in data.get('questions', []):
            # Normalize options to always be a list
            raw_options = mcq.get('options', [])
            if isinstance(raw_options, str):
                # LLM sometimes returns options as a single string - split it
                if '\n' in raw_options:
                    raw_options = [opt.strip() for opt in raw_options.split('\n') if opt.strip()]
                elif raw_options.count(',') >= 3:
                    # Try splitting by comma for "A: ..., B: ..., C: ..., D: ..."
                    raw_options = [opt.strip() for opt in raw_options.split(',') if opt.strip()]
                else:
                    raw_options = [raw_options]
            
            card = Card(
                paper_id,
                mcq['q'],
                mcq.get('answer', ''),
                card_type='mcq',
                options=raw_options
            )
            card.topic = extract_topic(title)
            cards.append(card)
        
        # Add flashcards
        for fc in data.get('flashcards', []):
            card = Card(
                paper_id,
                fc['q'],
                fc['a'],
                card_type='flashcard'
            )
            card.topic = extract_topic(title)
            cards.append(card)

        # Cap at 10 cards; if the LLM returned too few, top up from the abstract
        if len(cards) < 10:
            cards.extend(_cards_from_dict(paper_id, title, _fallback_cards(title, abstract)))
        return cards[:10]

    except Exception as e:
        print(f"Error generating cards: {e}")
        # Return abstract-derived cards so Study Mode is never empty
        return _cards_from_dict(paper_id, title, _fallback_cards(title, abstract))[:10]


def _split_sentences(text):
    """Lightweight sentence splitter for building fallback cards."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    return [p.strip() for p in parts if len(p.strip()) > 30]


def _fallback_cards(title, abstract):
    """
    Build up to 10 study cards from the abstract alone (no LLM required).
    Produces cloze-style MCQs plus concept flashcards so the deck is still useful offline.
    """
    sentences = _split_sentences(abstract)
    questions = []
    flashcards = [
        {'q': f"What is the paper '{title}' about?", 'a': (abstract[:240] + '…') if len(abstract) > 240 else abstract},
        {'q': "What is the single most important contribution?", 'a': sentences[0] if sentences else title},
    ]
    if len(sentences) > 1:
        flashcards.append({'q': "What problem or motivation does the paper describe?", 'a': sentences[0]})

    # Build cloze MCQs by blanking a salient keyword in each sentence
    import re
    stop = {'this', 'that', 'with', 'from', 'these', 'their', 'which', 'while', 'using',
            'based', 'such', 'than', 'then', 'they', 'have', 'been', 'into', 'also', 'more',
            'most', 'paper', 'propose', 'present', 'results', 'method', 'approach'}
    for sent in sentences:
        words = [w for w in re.findall(r'\b[A-Za-z][A-Za-z\-]{4,}\b', sent) if w.lower() not in stop]
        if not words:
            continue
        key = max(words, key=len)
        distractors = [w for w in ['architecture', 'baseline', 'benchmark', 'optimization',
                                    'generalization', 'representation', 'inference', 'alignment']
                       if w.lower() != key.lower()][:3]
        opts = [key] + distractors
        random.shuffle(opts)
        labels = ['A', 'B', 'C', 'D']
        answer = labels[opts.index(key)]
        blanked = re.sub(r'\b' + re.escape(key) + r'\b', '_____', sent, count=1)
        questions.append({
            'q': f"Fill in the blank: {blanked}",
            'options': [f"{labels[i]}) {opts[i]}" for i in range(4)],
            'answer': answer,
        })
        if len(questions) >= 7:
            break

    if not questions:
        questions.append({
            'q': f"What is the main contribution of '{title}'?",
            'options': ['A) A novel method', 'B) A new dataset', 'C) A benchmark', 'D) An analysis study'],
            'answer': 'A',
        })
    return {'questions': questions, 'flashcards': flashcards}


def _cards_from_dict(paper_id, title, data):
    """Convert a {'questions': [...], 'flashcards': [...]} dict into Card objects."""
    out = []
    for mcq in data.get('questions', []):
        opts = mcq.get('options', [])
        if isinstance(opts, str):
            opts = [o.strip() for o in opts.split('\n') if o.strip()] or [opts]
        c = Card(paper_id, mcq['q'], mcq.get('answer', ''), card_type='mcq', options=opts)
        c.topic = extract_topic(title)
        out.append(c)
    for fc in data.get('flashcards', []):
        c = Card(paper_id, fc['q'], fc['a'], card_type='flashcard')
        c.topic = extract_topic(title)
        out.append(c)
    return out


def extract_topic(title):
    """Extract research topic from paper title."""
    keywords = {
        'Large Language Models': ['llm', 'language model', 'transformer'],
        'Computer Vision': ['vision', 'image', 'visual'],
        'NLP': ['nlp', 'text', 'language'],
        'Reinforcement Learning': ['reinforcement', 'rl', 'agent']
    }
    
    title_lower = title.lower()
    for topic, words in keywords.items():
        if any(w in title_lower for w in words):
            return topic
    
    return 'General ML'


def save_cards(cards, filepath=STUDY_FILE):
    """Save cards to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for card in cards:
            f.write(json.dumps(card.to_dict()) + '\n')


def load_cards(filepath=STUDY_FILE):
    """Load cards from JSONL file."""
    if not os.path.exists(filepath):
        return []
    
    cards = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    cards.append(Card.from_dict(data))
                except:
                    continue
    
    return cards


def review_card(card, quality, save_cards_fn=None):
    """
    Review a card and update scheduling.
    
    Args:
        card: Card to review
        quality: 0-5 quality rating
        save_cards_fn: Optional callback to save cards
    
    Returns:
        Updated card
    """
    # Update SM-2 scheduling
    SM2Scheduler.update_card(card, quality)
    
    # Update RL parameters
    correct = quality >= 3
    RLCardScheduler.update_card_rl(card, correct)
    
    return card
