"""
Test suite for new Scholar Inbox features:
- Gap Finder
- Citation Finder
- Paper Comparison
- Study Mode
"""

import unittest
import numpy as np
from core.gap_finder import compute_novelty_score, find_sparse_regions
from core.citation_finder import split_into_sentences
from core.study import Card, SM2Scheduler, RLCardScheduler
from agents.compare_agent import ComparisonAgent

class TestGapFinder(unittest.TestCase):
    
    def test_novelty_score(self):
        """Test novelty score computation"""
        # Create sample embeddings
        embedding = np.array([1, 0, 0])
        all_embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        score = compute_novelty_score(embedding, all_embeddings)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_sparse_regions(self):
        """Test sparse region detection"""
        coords = np.random.randn(50, 2)
        embeddings = np.random.randn(50, 10)
        
        regions = find_sparse_regions(embeddings, coords, num_regions=3)
        self.assertIsInstance(regions, list)


class TestCitationFinder(unittest.TestCase):
    
    def test_sentence_splitting(self):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_into_sentences(text)
        
        self.assertEqual(len(sentences), 3)
        self.assertIn("First sentence", sentences[0])


class TestStudyMode(unittest.TestCase):
    
    def test_card_creation(self):
        """Test study card creation"""
        card = Card('paper_123', 'What is AI?', 'Artificial Intelligence')
        
        self.assertEqual(card.paper_id, 'paper_123')
        self.assertEqual(card.question, 'What is AI?')
        self.assertEqual(card.answer, 'Artificial Intelligence')
    
    def test_sm2_update(self):
        """Test SM-2 algorithm updates"""
        card = Card('p1', 'Q', 'A')
        initial_ease = card.ease_factor
        
        # Correct answer (quality=4)
        SM2Scheduler.update_card(card, 4)
        
        self.assertEqual(card.repetitions, 1)
        self.assertGreater(card.ease_factor, initial_ease)
    
    def test_rl_sampling(self):
        """Test Thompson Sampling for card selection"""
        cards = [
            Card(f'p{i}', f'Q{i}', f'A{i}') for i in range(5)
        ]
        
        for card in cards:
            card.alpha = 5
            card.beta = 2
        
        selected = RLCardScheduler.select_next_card(cards)
        self.assertIn(selected, cards)


class TestComparison(unittest.TestCase):
    
    def test_comparison_agent_init(self):
        """Test comparison agent initialization"""
        agent = ComparisonAgent('test_key', use_groq=False)
        
        self.assertIsNotNone(agent)
        self.assertEqual(len(agent.COMPARISON_FIELDS), 7)


if __name__ == '__main__':
    unittest.main()
