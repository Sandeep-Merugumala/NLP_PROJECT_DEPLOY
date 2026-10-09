"""
Study Mode with Thompson Sampling + SM-2 Spaced Repetition
Part 5 Feature with Reinforcement Learning Integration
[FIXED VERSION - All bugs corrected]

The RL agent learns which cards to prioritize based on user performance,
using Thompson Sampling (a Bayesian bandit algorithm) over SM-2 baseline.
"""

import json
import uuid
import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardDifficulty(Enum):
    """SM-2 difficulty scale"""
    TRIVIAL = 0        # Too easy
    DIFFICULT = 1      # Hard
    ACCEPTABLE = 2     # Just right
    EASY = 3           # Easy but good
    TRIVIAL_EASY = 4   # Way too easy

@dataclass
class FlashCard:
    """Flashcard with spaced repetition state"""
    card_id: str
    paper_id: str
    question: str
    answer: str
    difficulty_factor: float = 2.5  # SM-2 easiness factor
    interval: int = 1               # Days until next review
    repetitions: int = 0            # Number of times reviewed
    next_review: str = None         # ISO timestamp
    creation_date: str = None
    last_review: str = None
    review_history: List[Dict] = None  # [{date, quality, new_interval, new_difficulty}]

@dataclass
class RLState:
    """State for RL agent"""
    card_id: str
    user_mastery: float  # 0-1 estimate of user's mastery
    card_difficulty: float  # 0-1 estimated difficulty
    time_since_review: int  # hours
    topic_mastery: float  # 0-1 mastery of card's topic
    session_performance: float  # 0-1 avg performance this session

@dataclass
class RLReward:
    """Reward signal for RL agent"""
    card_id: str
    timestamp: str
    user_response_quality: int  # 0-4 (SM-2 quality)
    was_correct: bool
    time_to_answer: float  # seconds
    base_reward: float  # SM-2 derived reward
    rl_reward: float  # RL-computed reward
    reward_reason: str

class ThompsonSamplingBandit:
    """Thompson Sampling bandit for card selection"""
    
    def __init__(self, num_arms: int, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize bandit with num_arms (cards).
        alpha, beta: Beta distribution hyperparameters.
        """
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms) * alpha  # Success counts
        self.beta = np.ones(num_arms) * beta    # Failure counts
    
    def select_arm(self) -> int:
        """
        Thompson Sampling: sample from each arm's Beta posterior,
        return arm with highest sample value.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """Update arm with binary outcome (0 or 1)"""
        if arm < 0 or arm >= self.num_arms:
            logger.warning(f"Invalid arm index {arm}, ignoring update")
            return
        
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_posterior_probs(self) -> np.ndarray:
        """Get posterior probabilities for each arm"""
        means = self.alpha / (self.alpha + self.beta)
        return means / means.sum()

class SM2SpacedRepetition:
    """
    SM-2 (SuperMemo 2) algorithm for spaced repetition.
    Classic baseline for comparing against RL.
    """
    
    def __init__(self):
        self.INITIAL_EF = 2.5  # Easiness factor
        self.INITIAL_INTERVAL = 1  # Days
    
    def calculate_next_interval(
        self,
        repetitions: int,
        quality: int,  # 0-4
        interval: int,
        difficulty_factor: float
    ) -> Tuple[int, float, int]:
        """
        Calculate next review interval using SM-2.
        Returns: (new_interval, new_difficulty_factor, new_repetitions)
        
        quality: 0-2 = fail, 3-4 = pass
        """
        
        if quality < 3:
            # Repeat immediately on failure
            return 1, difficulty_factor, 0
        
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 3
        else:
            new_interval = round(interval * difficulty_factor)
        
        # Update difficulty factor
        new_ef = difficulty_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ef = max(1.3, new_ef)  # EF must be >= 1.3
        
        return new_interval, new_ef, repetitions + 1
    
    def get_next_review_time(self, interval_days: int) -> str:
        """Get ISO timestamp for next review"""
        next_review = datetime.now() + timedelta(days=interval_days)
        return next_review.isoformat()

class StudyModeRL:
    """Study Mode with RL-powered card selection"""
    
    def __init__(
        self,
        llm_router,
        rag_pipeline,
        use_rl: bool = True,
        storage_path: str = "data/study_rl.jsonl"
    ):
        self.llm = llm_router
        self.rag = rag_pipeline
        self.use_rl = use_rl
        self.storage_path = storage_path
        
        self.cards = {}  # card_id -> FlashCard
        self.sessions = {}  # session_id -> {cards_reviewed, rewards, ...}
        self.bandits = {}  # paper_id -> ThompsonSamplingBandit
        self.sm2 = SM2SpacedRepetition()
        
        # BUG FIX #9: Maintain stable card-to-arm mapping (per paper)
        self.card_id_to_arm = {}  # card_id -> arm_index
        
        # Create storage directory
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self._load_cards()
    
    def generate_cards_from_paper(
        self,
        paper_id: str,
        paper_data: Dict,
        num_cards: int = 5
    ) -> List[FlashCard]:
        """
        Generate flashcards from a paper using LLM.
        Creates both factual and conceptual questions.
        """
        
        abstract = paper_data.get("abstract", "")
        title = paper_data.get("title", "")
        
        # Retrieve key sections from RAG
        chunks = self.rag.retrieve(paper_id, top_k=3)
        content = "\n\n".join([c["content"] for c in chunks])
        
        prompt = f"""
Generate {num_cards} flashcards for studying this research paper.
Mix factual questions (specific findings, methodology) with conceptual questions (big ideas, implications).

Title: {title}
Abstract: {abstract}
Content: {content[:2000]}

For each card, create:
1. A clear, specific question
2. A concise, accurate answer (2-3 sentences)
3. Difficulty estimate (0-1, where 1 is hardest)

Return as JSON:
{{
  "cards": [
    {{
      "question": "What is...",
      "answer": "...",
      "type": "factual" or "conceptual",
      "difficulty": 0.5
    }}
  ]
}}
"""
        
        response = self.llm.generate(
            prompt,
            temperature=0.7,
            max_tokens=2000,
            format_json=True
        )
        
        try:
            card_data = json.loads(response)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse card generation response: {e}, using fallback")
            # Fallback
            card_data = {"cards": []}
        
        cards = []
        existing_arms = [
            self.card_id_to_arm[c.card_id]
            for c in self.cards.values()
            if c.paper_id == paper_id and c.card_id in self.card_id_to_arm
        ]
        next_arm_index = (max(existing_arms) + 1) if existing_arms else 0

        for card_spec in card_data.get("cards", [])[:num_cards]:
            card_id = str(uuid.uuid4())[:12]
            card = FlashCard(
                card_id=card_id,
                paper_id=paper_id,
                question=card_spec.get("question", ""),
                answer=card_spec.get("answer", ""),
                difficulty_factor=2.5,
                interval=1,
                repetitions=0,
                next_review=datetime.now().isoformat(),
                creation_date=datetime.now().isoformat(),
                last_review=None,
                review_history=[]
            )
            self.cards[card_id] = card
            
            # BUG FIX #9: Map card_id to arm_index
            self.card_id_to_arm[card_id] = next_arm_index
            next_arm_index += 1
            
            cards.append(card)
        
        # Initialize bandit for this paper if not exists
        if paper_id not in self.bandits and cards:
            required_arms = self._required_arms_for_paper(paper_id)
            self.bandits[paper_id] = ThompsonSamplingBandit(
                num_arms=required_arms,
                alpha=1.0,
                beta=1.0
            )
        elif cards:
            self._ensure_bandit_capacity(paper_id)
        
        self._save_cards()
        return cards
    
    def start_study_session(
        self,
        paper_id: str,
        num_cards: int = 5
    ) -> Dict:
        """Start a study session for a paper"""
        
        session_id = str(uuid.uuid4())[:8]
        
        # Get cards due for review
        due_cards = self._get_due_cards(paper_id, limit=num_cards)
        
        if not due_cards:
            # No cards due, show earliest created
            paper_cards = [c for c in self.cards.values() if c.paper_id == paper_id]
            due_cards = sorted(paper_cards, key=lambda c: c.creation_date or "")[:num_cards]
        
        session = {
            "session_id": session_id,
            "paper_id": paper_id,
            "started_at": datetime.now().isoformat(),
            "cards_to_review": [c.card_id for c in due_cards],
            "current_card_index": 0,
            "cards_reviewed": [],
            "rewards": [],
            "total_score": 0.0,
            "use_rl": self.use_rl
        }
        
        self.sessions[session_id] = session
        return session
    
    def get_next_card(
        self,
        session_id: str
    ) -> Optional[Dict]:
        """
        Get next card to review, using RL card selection if enabled.
        
        If RL is enabled: Thompson Sampling bandit selects the card.
        If RL is disabled: SM-2 or random ordering.
        """
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        cards_to_review = session["cards_to_review"]
        current_idx = session["current_card_index"]
        
        if current_idx >= len(cards_to_review):
            return None  # Session complete
        
        if self.use_rl:
            # RL-based card selection: Thompson Sampling
            next_card_id = self._select_card_rl(session)
        else:
            # Baseline: SM-2 ordering (already sorted)
            next_card_id = cards_to_review[current_idx]
        
        card = self.cards.get(next_card_id)
        if not card:
            return None
        
        session["current_card_index"] = current_idx + 1
        
        return {
            "card_id": card.card_id,
            "question": card.question,
            "paper_id": card.paper_id,
            "card_number": current_idx + 1,
            "total_cards": len(cards_to_review)
        }
    
    def submit_response(
        self,
        session_id: str,
        card_id: str,
        user_answer: str,
        quality: int,  # 0-4 (SM-2 quality scale)
        time_to_answer: float  # seconds
    ) -> Dict:
        """
        Process user's response and compute rewards.
        
        Args:
            quality: 0-2 = incorrect/hard, 3-4 = correct/easy
            time_to_answer: Time spent thinking (seconds)
        
        Returns:
            {
                "base_reward": float (SM-2 based),
                "rl_reward": float (RL bonus/penalty),
                "total_reward": float,
                "new_interval": int (days),
                "feedback": str
            }
        """
        
        if session_id not in self.sessions or card_id not in self.cards:
            logger.warning(f"Invalid session {session_id} or card {card_id}")
            return {}
        
        session = self.sessions[session_id]
        card = self.cards[card_id]
        
        # 1. Compute SM-2 update
        new_interval, new_ef, new_reps = self.sm2.calculate_next_interval(
            card.repetitions,
            quality,
            card.interval,
            card.difficulty_factor
        )
        
        # 2. Compute base reward (from SM-2)
        base_reward = self._compute_base_reward(quality, new_interval, card.difficulty_factor)
        
        # 3. Compute RL reward (bonus for good difficulty calibration)
        rl_reward = 0.0
        reward_reason = ""
        
        if self.use_rl:
            rl_reward, reward_reason = self._compute_rl_reward(
                card, quality, time_to_answer, new_interval
            )
        
        total_reward = base_reward + rl_reward
        
        # 4. Update card state
        card.last_review = datetime.now().isoformat()
        card.next_review = self.sm2.get_next_review_time(new_interval)
        card.interval = new_interval
        card.difficulty_factor = new_ef
        card.repetitions = new_reps
        
        review_entry = {
            "date": datetime.now().isoformat(),
            "quality": quality,
            "new_interval": new_interval,
            "new_difficulty": new_ef,
            "time_to_answer": time_to_answer,
            "reward": total_reward
        }
        if card.review_history is None:
            card.review_history = []
        card.review_history.append(review_entry)
        
        # 5. Update session
        session["cards_reviewed"].append(card_id)
        session["rewards"].append(total_reward)
        session["total_score"] += total_reward
        
        # 6. Update bandit (RL learning)
        # BUG FIX #9: Use stable card_id_to_arm mapping
        if self.use_rl and session["paper_id"] in self.bandits:
            # Normalize quality to 0-1 reward
            normalized_reward = quality / 4.0
            bandit = self.bandits[session["paper_id"]]
            self._ensure_bandit_capacity(session["paper_id"])
            
            # Use stable mapping instead of list search
            if card_id in self.card_id_to_arm:
                arm_index = self.card_id_to_arm[card_id]
                bandit.update(arm_index, normalized_reward)
            else:
                logger.warning(f"Card {card_id} not in arm mapping")
        
        self._save_cards()
        
        return {
            "base_reward": round(base_reward, 3),
            "rl_reward": round(rl_reward, 3),
            "total_reward": round(total_reward, 3),
            "new_interval": new_interval,
            "new_repetitions": new_reps,
            "reward_reason": reward_reason,
            "feedback": self._generate_feedback(quality, time_to_answer)
        }
    
    def complete_session(self, session_id: str) -> Dict:
        """Complete a study session and return summary"""
        
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        summary = {
            "session_id": session_id,
            "duration_minutes": round(
                (datetime.now() - datetime.fromisoformat(session["started_at"])).total_seconds() / 60,
                1
            ),
            "cards_reviewed": len(session["cards_reviewed"]),
            "total_rewards": round(sum(session["rewards"]), 2) if session["rewards"] else 0.0,
            "average_reward": round(np.mean(session["rewards"]), 3) if session["rewards"] else 0.0,
            "performance_trend": self._analyze_performance_trend(session["rewards"]),
            "study_method": "RL (Thompson Sampling)" if self.use_rl else "SM-2 Baseline"
        }
        
        return summary
    
    # ============ RL Methods ============
    
    def _select_card_rl(self, session: Dict) -> str:
        """
        RL-based card selection using Thompson Sampling.
        """
        
        cards_to_review = session["cards_to_review"]
        paper_id = session["paper_id"]
        
        if paper_id not in self.bandits:
            required_arms = self._required_arms_for_paper(paper_id)
            self.bandits[paper_id] = ThompsonSamplingBandit(
                num_arms=max(required_arms, 1),
                alpha=1.0,
                beta=1.0
            )
        else:
            self._ensure_bandit_capacity(paper_id)
        
        bandit = self.bandits[paper_id]

        # Thompson Sampling over only cards available in this session
        arm_samples = np.random.beta(bandit.alpha, bandit.beta)
        available = []
        for card_id in cards_to_review:
            arm = self.card_id_to_arm.get(card_id)
            if arm is not None and 0 <= arm < bandit.num_arms:
                available.append((card_id, arm))

        if not available:
            return cards_to_review[0]

        best_card_id, _ = max(available, key=lambda pair: arm_samples[pair[1]])
        return best_card_id

    def _required_arms_for_paper(self, paper_id: str) -> int:
        """Compute number of required bandit arms for a paper."""
        paper_arms = [
            self.card_id_to_arm[card.card_id]
            for card in self.cards.values()
            if card.paper_id == paper_id and card.card_id in self.card_id_to_arm
        ]
        return (max(paper_arms) + 1) if paper_arms else 0

    def _ensure_bandit_capacity(self, paper_id: str):
        """Grow an existing bandit so all mapped arms are valid."""
        if paper_id not in self.bandits:
            return

        required = self._required_arms_for_paper(paper_id)
        bandit = self.bandits[paper_id]

        if required <= bandit.num_arms:
            return

        extra = required - bandit.num_arms
        bandit.alpha = np.concatenate([bandit.alpha, np.ones(extra)])
        bandit.beta = np.concatenate([bandit.beta, np.ones(extra)])
        bandit.num_arms = required
    
    def _compute_base_reward(
        self,
        quality: int,
        new_interval: int,
        difficulty_factor: float
    ) -> float:
        """Compute SM-2 derived reward"""
        
        # Reward components:
        # 1. Quality: Higher quality = higher reward
        quality_bonus = quality / 4.0
        
        # 2. Interval growth: Longer intervals = success = reward
        interval_bonus = min(new_interval / 30.0, 1.0)  # Cap at 1 month
        
        # 3. Difficulty calibration: EF > 2.5 = good
        difficulty_bonus = max(0, (difficulty_factor - 2.5) / 1.0) * 0.1
        
        return quality_bonus * 0.7 + interval_bonus * 0.2 + difficulty_bonus * 0.1
    
    def _compute_rl_reward(
        self,
        card: FlashCard,
        quality: int,
        time_to_answer: float,
        new_interval: int
    ) -> Tuple[float, str]:
        """
        Compute RL reward: bonus for:
        - Correct answer at "sweet spot" difficulty
        - Time spent indicates engagement
        - Growing intervals show learning
        """
        
        rl_reward = 0.0
        reason = ""
        
        # 1. Difficulty sweet spot: quality 3 or 4 is ideal (user is challenged but capable)
        if quality == 4:
            rl_reward += 0.1
            reason += "Easy mastery bonus. "
        elif quality == 3:
            rl_reward += 0.15
            reason += "Perfect difficulty sweet spot. "
        elif quality == 2:
            rl_reward -= 0.1
            reason += "Too difficult, needs review. "
        
        # 2. Engagement: moderate thinking time is good (30s - 120s)
        if 30 <= time_to_answer <= 120:
            rl_reward += 0.05
            reason += "Good engagement time. "
        elif time_to_answer < 10:
            rl_reward -= 0.05
            reason += "Too quick, possibly careless. "
        
        # 3. Learning progress: longer intervals indicate mastery
        if new_interval >= 7:
            rl_reward += 0.05
            reason += "Strong learning progress. "
        
        return rl_reward, reason
    
    def _get_due_cards(self, paper_id: str, limit: int = 10) -> List[FlashCard]:
        """Get cards due for review
        
        BUG FIX #10: Handle None next_review values and invalid ISO formats
        """
        
        now = datetime.now()
        due = []
        
        for card in self.cards.values():
            if card.paper_id != paper_id:
                continue
            
            # BUG FIX: Check if next_review exists and is valid
            if not card.next_review:
                logger.warning(f"Card {card.card_id} has no next_review, skipping")
                continue
            
            try:
                next_review = datetime.fromisoformat(card.next_review)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid ISO format for card {card.card_id}: {card.next_review} - {e}")
                continue
            
            if next_review <= now:
                due.append(card)
        
        # Sort: earlier due dates first, then by difficulty (harder first)
        due.sort(key=lambda c: (
            datetime.fromisoformat(c.next_review),
            -c.difficulty_factor
        ))
        
        return due[:limit]
    
    def _analyze_performance_trend(self, rewards: List[float]) -> str:
        """Analyze if performance is improving"""
        
        if len(rewards) < 2:
            return "Insufficient data"
        
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        
        improvement = second_half - first_half
        
        if improvement > 0.1:
            return "Improving 📈"
        elif improvement < -0.1:
            return "Declining 📉"
        else:
            return "Stable ➡️"
    
    def _generate_feedback(self, quality: int, time_to_answer: float) -> str:
        """Generate encouraging feedback"""
        
        if quality >= 4:
            return "🎉 Excellent! You've mastered this."
        elif quality == 3:
            return "✅ Good! You understand this well."
        elif quality == 2:
            return "⚠️ This needs more practice. Don't worry!"
        else:
            return "💪 Let's focus on this. You'll get it next time."
    
    def _load_cards(self):
        """Load cards from storage
        
        BUG FIX #12: Specific exception handling
        """
        
        if not os.path.exists(self.storage_path):
            logger.info(f"Card storage file does not exist yet: {self.storage_path}")
            return
        
        try:
            paper_arm_counts = defaultdict(int)
            with open(self.storage_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        card_data = json.loads(line)
                        card = FlashCard(**card_data)
                        self.cards[card.card_id] = card
                        # Rebuild stable per-paper arm mapping
                        self.card_id_to_arm[card.card_id] = paper_arm_counts[card.paper_id]
                        paper_arm_counts[card.paper_id] += 1
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse card: {e}")
                        continue
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Error loading cards, starting with empty storage: {e}")
    
    def _save_cards(self):
        """Persist cards to storage
        
        BUG FIX #11: Ensure directory exists before writing
        """
        
        # Ensure directory exists
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.storage_path, 'w') as f:
                for card in self.cards.values():
                    f.write(json.dumps(asdict(card), default=str) + "\n")
        except IOError as e:
            logger.error(f"Failed to save cards: {e}")
    
    def get_study_stats(self, paper_id: str) -> Dict:
        """Get statistics for a paper's study progress"""
        
        paper_cards = [c for c in self.cards.values() if c.paper_id == paper_id]
        
        if not paper_cards:
            return {}
        
        total_reviews = sum(len(c.review_history or []) for c in paper_cards)
        avg_difficulty = np.mean([c.difficulty_factor for c in paper_cards])
        avg_interval = np.mean([c.interval for c in paper_cards])
        
        # BUG FIX: Avoid division by zero for mastery estimate
        mastery_scores = []
        for c in paper_cards:
            if c.repetitions > 0:
                mastery_scores.append(min(c.repetitions / 5, 1.0))
            else:
                mastery_scores.append(0.0)
        
        mastery_estimate = np.mean(mastery_scores) if mastery_scores else 0.0
        
        return {
            "paper_id": paper_id,
            "num_cards": len(paper_cards),
            "total_reviews": total_reviews,
            "avg_difficulty_factor": round(avg_difficulty, 2),
            "avg_interval_days": round(avg_interval, 1),
            "mastery_estimate": round(mastery_estimate, 2)
        }
