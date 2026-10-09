"""
Citation Finder / Related Work Writer
Finds relevant papers for text sections and generates citation suggestions
with BibTeX export and LLM-rewritten paragraphs.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .llm_helper import call_llm_with_retry

def split_into_sentences(text):
    """Split text into sentences, preserving structure."""
    # Simple sentence splitter using period, exclamation, question mark
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def find_related_papers(sentence, papers_db, tfidf_vectorizer, tfidf_matrix, top_k=3):
    """
    Find most relevant papers for a given sentence using TF-IDF similarity.
    
    Returns:
        List of (paper_dict, similarity_score) tuples
    """
    if not papers_db or tfidf_matrix.shape[0] == 0:
        return []
    
    try:
        # Vectorize the sentence
        sentence_vector = tfidf_vectorizer.transform([sentence])
        
        # Compute similarities
        similarities = cosine_similarity(sentence_vector, tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return papers and scores
        results = []
        for idx in top_indices:
            if 0 <= idx < len(papers_db):
                results.append((papers_db[idx], float(similarities[idx])))
        
        return results
    
    except Exception as e:
        print(f"Error in find_related_papers: {e}")
        return []

def generate_bibtex(paper):
    """
    Generate BibTeX entry for a paper.
    """
    paper_id = paper.get('id', 'unknown')
    title = paper.get('title', 'Unknown').replace('{', '\\{').replace('}', '\\}')
    authors = paper.get('authors', 'Unknown')
    year = paper.get('published_date', '2024')[:4]
    
    # Extract year from date string if needed
    if len(year) != 4 or not year.isdigit():
        year = '2024'
    
    # Simplify authors (take first author + et al)
    if isinstance(authors, list):
        author_str = authors[0] if authors else 'Unknown'
    else:
        author_str = authors if authors else 'Unknown'
    
    arxiv_id = paper_id.replace('/', '.')
    
    bibtex = f"""@article{{{arxiv_id},
  title={{{title}}},
  author={{{author_str}}},
  journal={{arXiv preprint arXiv:{paper_id}}},
  year={{{year}}}
}}"""
    
    return bibtex

def suggest_citations_for_paragraph(paragraph_text, papers_db, tfidf_vectorizer, tfidf_matrix, top_k=3):
    """
    Suggest citations for a paragraph by analyzing each sentence.
    
    Returns:
        {
            'sentences': [
                {
                    'text': str,
                    'suggested_papers': [
                        {'title': str, 'id': str, 'authors': list, 'score': float}
                    ]
                },
                ...
            ],
            'all_papers': {...},  # Deduplicated
            'bibtex': str
        }
    """
    sentences = split_into_sentences(paragraph_text)
    
    result = {
        'sentences': [],
        'all_papers': {},
        'bibtex_entries': []
    }
    
    seen_papers = set()
    
    for sentence in sentences:
        related = find_related_papers(sentence, papers_db, tfidf_vectorizer, tfidf_matrix, top_k)
        
        sentence_result = {
            'text': sentence,
            'suggested_papers': []
        }
        
        for paper, score in related:
            paper_id = paper.get('id', '')
            
            paper_summary = {
                'id': paper_id,
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', []),
                'similarity_score': float(score),
                'abstract': paper.get('abstract', '')[:200]
            }
            
            sentence_result['suggested_papers'].append(paper_summary)
            
            if paper_id not in seen_papers:
                seen_papers.add(paper_id)
                result['all_papers'][paper_id] = paper_summary
                result['bibtex_entries'].append(generate_bibtex(paper))
        
        result['sentences'].append(sentence_result)
    
    result['bibtex'] = '\n\n'.join(result['bibtex_entries'])
    
    return result

def rewrite_paragraph_with_citations(paragraph_text, citation_suggestions, api_key, use_groq=False):
    """
    Use LLM to rewrite paragraph with natural citation integration.
    
    Returns:
        {
            'original': str,
            'rewritten': str,
            'citations_added': int
        }
    """
    # Format cited papers
    papers_text = "\n".join([
        f"- [{p['id']}] {p['title']} ({', '.join(p['authors'][:2])})"
        for p in citation_suggestions['all_papers'].values()
    ])
    
    if not papers_text:
        return {
            'original': paragraph_text,
            'rewritten': paragraph_text,
            'citations_added': 0
        }
    
    prompt = f"""Rewrite this paragraph to naturally incorporate citations to the relevant papers.
Use [arXiv_id] format for citations. Keep the original meaning and flow.

PARAGRAPH:
{paragraph_text}

RELEVANT PAPERS:
{papers_text}

REQUIREMENTS:
1. Add citations where relevant
2. Keep the original message and tone
3. Don't add more than one citation per sentence
4. Format citations as [arxiv_id]
5. Return ONLY the rewritten paragraph, no explanations"""
    
    try:
        rewritten = call_llm_with_retry(prompt, api_key, use_groq=use_groq)
        
        # Count added citations
        citations_added = len(re.findall(r'\[\d+\.\d+\]', rewritten))
        
        return {
            'original': paragraph_text,
            'rewritten': rewritten if rewritten else paragraph_text,
            'citations_added': citations_added,
            'papers_suggested': len(citation_suggestions['all_papers'])
        }
    
    except Exception as e:
        print(f"Error rewriting paragraph: {e}")
        return {
            'original': paragraph_text,
            'rewritten': paragraph_text,
            'citations_added': 0,
            'error': str(e)
        }

def process_citation_request(user_text, papers_db, tfidf_vectorizer, tfidf_matrix, api_key, use_groq=False):
    """
    Main entry point: process user text for citation suggestions.
    
    Returns:
        {
            'citations': [...],
            'rewritten_paragraph': str,
            'bibtex': str,
            'statistics': {'papers_found': int, 'citations_suggested': int}
        }
    """
    # Get citation suggestions
    suggestions = suggest_citations_for_paragraph(user_text, papers_db, tfidf_vectorizer, tfidf_matrix, top_k=5)
    
    # Rewrite with citations
    rewrite_result = rewrite_paragraph_with_citations(user_text, suggestions, api_key, use_groq)
    
    return {
        'original_text': user_text,
        'citations': suggestions['sentences'],
        'rewritten_paragraph': rewrite_result['rewritten'],
        'bibtex': suggestions['bibtex'],
        'statistics': {
            'papers_found': len(suggestions['all_papers']),
            'citations_suggested': rewrite_result['citations_added'],
            'sentences_analyzed': len(suggestions['sentences'])
        }
    }
