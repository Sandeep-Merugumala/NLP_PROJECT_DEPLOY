"""
Paper Comparison Studio
LangGraph agent for extracting structured information from papers
and generating side-by-side comparison tables.
"""

import json
from core.llm_helper import call_llm_with_retry

class ComparisonAgent:
    """Agent for comparing multiple papers side-by-side."""
    
    COMPARISON_FIELDS = [
        'Problem/Research Question',
        'Methodology',
        'Dataset/Benchmark',
        'Key Results',
        'Limitations',
        'Year',
        'Venue/Conference'
    ]
    
    def __init__(self, api_key, use_groq=False):
        self.api_key = api_key
        self.use_groq = use_groq
    
    def extract_paper_info(self, paper):
        """
        Extract structured information from a paper using LLM.
        
        Args:
            paper: Dict with 'title', 'abstract', 'id', etc.
        
        Returns:
            Dict with extracted fields
        """
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        year = paper.get('published', '2024')[:4]
        
        prompt = f"""Extract the following information from this paper:

TITLE: {title}
ABSTRACT: {abstract}

Extract these fields for the paper. Be concise (1-2 sentences each):
1. Problem/Research Question - What problem does this paper solve?
2. Methodology - What approach/method is used?
3. Dataset/Benchmark - What datasets or benchmarks are used?
4. Key Results - What are the main findings/results?
5. Limitations - What are the main limitations?
6. Venue - Is this from a specific venue/conference?

Return as JSON only with keys: problem, methodology, dataset, results, limitations, venue"""
        
        try:
            response = call_llm_with_retry(prompt, self.api_key, use_groq=self.use_groq)
            
            # Parse JSON response
            if isinstance(response, str):
                # Try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(response)
            else:
                data = response
            
            return {
                'paper_id': paper.get('id', ''),
                'title': title,
                'year': year,
                'problem': data.get('problem', 'N/A'),
                'methodology': data.get('methodology', 'N/A'),
                'dataset': data.get('dataset', 'N/A'),
                'results': data.get('results', 'N/A'),
                'limitations': data.get('limitations', 'N/A'),
                'venue': data.get('venue', 'arXiv')
            }
        
        except Exception as e:
            print(f"Error extracting paper info: {e}")
            # Return defaults
            return {
                'paper_id': paper.get('id', ''),
                'title': title,
                'year': year,
                'problem': abstract[:100] + '...',
                'methodology': 'See abstract',
                'dataset': 'Not specified',
                'results': 'See abstract',
                'limitations': 'Not specified',
                'venue': 'arXiv'
            }
    
    def compare_papers(self, paper_list):
        """
        Compare multiple papers and extract structured comparison.
        
        Args:
            paper_list: List of paper dicts
        
        Returns:
            {
                'comparison_table': {
                    'problem': [...],
                    'methodology': [...],
                    ...
                },
                'papers': [...],
                'when_to_use': str,
                'similarity_score': float
            }
        """
        if not paper_list:
            return {'error': 'No papers to compare', 'comparison_table': {}}
        
        # Extract info from each paper
        papers_info = []
        for paper in paper_list[:3]:  # Limit to 3 papers for clarity
            info = self.extract_paper_info(paper)
            papers_info.append(info)
        
        # Build comparison table
        comparison_table = {
            'Paper': [p['title'][:50] + '...' if len(p['title']) > 50 else p['title'] for p in papers_info],
            'Year': [p['year'] for p in papers_info],
            'Problem': [p['problem'] for p in papers_info],
            'Methodology': [p['methodology'] for p in papers_info],
            'Dataset': [p['dataset'] for p in papers_info],
            'Results': [p['results'] for p in papers_info],
            'Limitations': [p['limitations'] for p in papers_info],
            'Venue': [p['venue'] for p in papers_info]
        }
        
        # Generate "when to use" recommendations
        when_to_use = self._generate_when_to_use(papers_info)
        
        # Calculate similarity (simple: shared keywords)
        similarity = self._calculate_similarity(papers_info)
        
        return {
            'comparison_table': comparison_table,
            'papers': papers_info,
            'when_to_use': when_to_use,
            'similarity_score': similarity,
            'count': len(papers_info)
        }
    
    def _generate_when_to_use(self, papers_info):
        """Generate recommendations for when to use each paper."""
        if len(papers_info) < 2:
            return "Single paper - review abstract and methodology."
        
        prompt = f"""Given these papers, provide 1-2 sentence recommendations for WHEN TO USE each one:

Papers:
{json.dumps(papers_info, indent=2)}

Format as JSON: {{"recommendations": ["...", "..."]}}"""
        
        try:
            response = call_llm_with_retry(prompt, self.api_key, use_groq=self.use_groq)
            
            if isinstance(response, str):
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(response)
                recs = data.get('recommendations', [])
                return " | ".join(recs[:len(papers_info)])
            
            return "Review papers for contextual fit."
        
        except Exception as e:
            return "Use based on your specific research goals and dataset availability."
    
    def _calculate_similarity(self, papers_info):
        """Simple similarity based on methodology overlap."""
        if len(papers_info) < 2:
            return 1.0
        
        # Simple keyword overlap in methodology
        methods = [p.get('methodology', '').lower() for p in papers_info]
        keywords = set()
        for method in methods:
            keywords.update(method.split())
        
        # Count shared keywords
        if not keywords:
            return 0.5
        
        overlap = 0
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                shared = len(set(method1.split()) & set(method2.split()))
                overlap += shared
        
        max_possible = len(methods) * len(keywords)
        return min(1.0, overlap / max_possible) if max_possible > 0 else 0.5
    
    def run(self, paper_ids, papers_db):
        """
        Main entry point: run comparison agent on given papers.
        
        Args:
            paper_ids: List of paper IDs to compare
            papers_db: Database of all papers
        
        Returns:
            Comparison result
        """
        # Find papers by ID
        papers_to_compare = [
            p for p in papers_db if p.get('id') in paper_ids
        ]
        
        if not papers_to_compare:
            return {'error': 'No papers found to compare', 'comparison_table': {}}
        
        return self.compare_papers(papers_to_compare)


def create_comparison_agent(api_key, use_groq=False):
    """Factory function to create a comparison agent."""
    return ComparisonAgent(api_key, use_groq)


def compare_papers_node(state, api_key, use_groq=False):
    """
    LangGraph node for paper comparison.
    
    State should contain:
        'paper_ids': List of paper IDs
        'papers_db': Paper database
    """
    agent = ComparisonAgent(api_key, use_groq)
    
    result = agent.run(
        state.get('paper_ids', []),
        state.get('papers_db', [])
    )
    
    return {
        'comparison_result': result,
        'status': 'complete'
    }
