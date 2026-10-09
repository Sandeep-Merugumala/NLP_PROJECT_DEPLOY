"""
Test suite for Research Trends Explorer
"""

import unittest
from core.trends import compute_trends, extract_topic_from_title, generate_trend_summary

class TestTrends(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.papers = [
            {
                'id': '2024.01.001',
                'title': 'Large Language Model Improvements',
                'abstract': 'We improve LLM performance',
                'published_date': '2024-01-15'
            },
            {
                'id': '2024.01.002',
                'title': 'Vision Transformer Networks',
                'abstract': 'Novel vision architecture',
                'published_date': '2024-01-20'
            },
            {
                'id': '2024.02.001',
                'title': 'Advanced LLM Fine-tuning',
                'abstract': 'Fine-tune language models',
                'published_date': '2024-02-10'
            }
        ]
    
    def test_compute_trends(self):
        """Test trend computation"""
        trends = compute_trends(self.papers)
        
        self.assertIn('trend_data', trends)
        self.assertIn('rising_topics', trends)
        self.assertIn('top_topics', trends)
        self.assertTrue(len(trends['trend_data']) > 0)
    
    def test_extract_topic(self):
        """Test topic extraction from title"""
        title = 'Large Language Model Fine-tuning Strategies'
        topic = extract_topic_from_title(title)
        
        self.assertIsNotNone(topic)
        self.assertIsInstance(topic, str)
    
    def test_empty_papers(self):
        """Test with empty paper list"""
        trends = compute_trends([])
        
        self.assertEqual(len(trends['trend_data']), 0)
        self.assertEqual(len(trends['rising_topics']), 0)


if __name__ == '__main__':
    unittest.main()
