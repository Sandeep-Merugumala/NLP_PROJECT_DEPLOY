"""
Research Gap and Idea Finder
Identifies sparse regions in the research embedding space and generates
novel research ideas with supporting papers.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .llm_helper import call_llm_with_retry

def find_sparse_regions(embeddings, t_sne_coords, num_regions=5, sparsity_threshold=0.3):
    """
    Find sparse (low-density) regions in the t-SNE embedding space.
    
    Args:
        embeddings: (n_papers, embedding_dim) - paper embeddings
        t_sne_coords: (n_papers, 2) - 2D t-SNE coordinates
        num_regions: Number of sparse regions to find
        sparsity_threshold: Local density threshold for considering a region sparse
    
    Returns:
        List of sparse region centers: [{'x': float, 'y': float, 'paper_indices': [int, ...]}]
    """
    if len(t_sne_coords) < 10:
        return []
    
    # Compute local density (average distance to k nearest neighbors)
    k = max(3, len(t_sne_coords) // 20)
    distances = np.sqrt(np.sum((t_sne_coords[:, None] - t_sne_coords[None, :]) ** 2, axis=2))
    
    local_densities = []
    for i in range(len(t_sne_coords)):
        k_nearest = np.sort(distances[i])[1:k+1]  # Skip self
        local_densities.append(np.mean(k_nearest))
    
    local_densities = np.array(local_densities)
    density_threshold = np.percentile(local_densities, sparsity_threshold * 100)
    
    # Find sparse points
    sparse_indices = np.where(local_densities > density_threshold)[0]
    
    if len(sparse_indices) == 0:
        return []
    
    # Cluster sparse points to find region centers
    sparse_regions = []
    remaining = set(sparse_indices)
    
    while remaining and len(sparse_regions) < num_regions:
        # Pick densest sparse region
        densest_idx = min(remaining, key=lambda i: local_densities[i])
        
        # Find neighbors of this point
        neighbor_dist_threshold = np.percentile(distances[densest_idx], 20)
        neighbors = [j for j in remaining if distances[densest_idx][j] < neighbor_dist_threshold]
        
        if neighbors:
            region_center = np.mean(t_sne_coords[neighbors], axis=0)
            sparse_regions.append({
                'x': float(region_center[0]),
                'y': float(region_center[1]),
                'paper_indices': [int(n) for n in neighbors],
                'sparsity_score': float(np.mean([local_densities[i] for i in neighbors]))
            })
            remaining -= set(neighbors)
    
    return sparse_regions

def compute_novelty_score(paper_embedding, all_embeddings, top_k=5):
    """
    Compute novelty score for a paper: 1 - max_cosine_similarity to corpus.
    Lower = more novel (unique).
    """
    if len(all_embeddings) == 0:
        return 0.5
    
    # Reshape if single embedding
    if len(paper_embedding.shape) == 1:
        paper_embedding = paper_embedding.reshape(1, -1)
    
    similarities = cosine_similarity(paper_embedding, all_embeddings)[0]
    # Take top-k similarity and average
    top_sims = np.sort(similarities)[-min(top_k, len(similarities)):]
    max_similarity = np.mean(top_sims)
    
    novelty = 1.0 - max_similarity
    return max(0.0, min(1.0, novelty))  # Clamp to [0, 1]

def generate_idea_suggestions(sparse_region, papers_db, embeddings, api_key, use_groq=False):
    """
    Generate research ideas for a sparse region.
    
    Returns:
        {
            'ideas': [
                {'title': str, 'description': str, 'supporting_papers': [paper_ids]},
                ...
            ],
            'novelty_scores': [float, ...],
            'region_center': {'x': float, 'y': float}
        }
    """
    if not sparse_region or not sparse_region.get('paper_indices'):
        return {'ideas': [], 'novelty_scores': [], 'region_center': None}
    
    # Get papers on the boundary of the sparse region
    boundary_papers = []
    for idx in sparse_region['paper_indices'][:3]:
        if idx < len(papers_db):
            boundary_papers.append(papers_db[idx])
    
    if not boundary_papers:
        return {'ideas': [], 'novelty_scores': [], 'region_center': sparse_region}
    
    # Create prompt for idea generation
    paper_context = "\n".join([
        f"- {p.get('title', 'Unknown')}: {p.get('abstract', '')[:200]}"
        for p in boundary_papers
    ])
    
    prompt = f"""Given these papers at the boundary of a sparse research area:

{paper_context}

Generate 3 novel research ideas that combine or bridge these areas. 
For each idea, provide:
1. A concise title
2. A 1-2 sentence description
3. Why this gap is important

Format as JSON with keys: "ideas" (array of {{"title": str, "description": str}}), "importance": str"""
    
    try:
        response = call_llm_with_retry(prompt, api_key, use_groq=use_groq)
        
        # Try to parse JSON
        import json
        idea_data = None
        if response and response.strip():
            try:
                idea_data = json.loads(response)
            except Exception:
                # Try to extract JSON block if wrapped in markdown
                import re
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        idea_data = json.loads(match.group())
                    except Exception:
                        pass
        
        ideas = idea_data.get('ideas', []) if isinstance(idea_data, dict) else []
        
        # Fallback to heuristic ideas if LLM returned nothing
        if not ideas:
            ideas = _generate_heuristic_ideas(boundary_papers)
        
        # Calculate novelty scores for each idea
        novelty_scores = [
            compute_novelty_score(embeddings[idx], embeddings) 
            for idx in sparse_region['paper_indices'][:len(ideas)]
        ]
        # Pad if fewer scores than ideas
        while len(novelty_scores) < len(ideas):
            novelty_scores.append(0.5)
        
        return {
            'ideas': ideas[:3],
            'novelty_scores': [float(s) for s in novelty_scores[:3]],
            'region_center': {'x': float(sparse_region['x']), 'y': float(sparse_region['y'])},
            'supporting_papers': boundary_papers
        }
    
    except Exception as e:
        print(f"Error generating ideas: {e}")
        heuristic_ideas = _generate_heuristic_ideas(boundary_papers)
        return {
            'ideas': heuristic_ideas,
            'novelty_scores': [0.6, 0.5, 0.55][:len(heuristic_ideas)],
            'region_center': {'x': float(sparse_region.get('x', 0)), 'y': float(sparse_region.get('y', 0))},
            'supporting_papers': boundary_papers,
            'error': str(e)
        }


def _generate_heuristic_ideas(boundary_papers):
    """Generate basic research ideas from paper titles/abstracts without LLM."""
    if not boundary_papers:
        return []
    
    ideas = []
    topics = [p.get('title', 'Unknown')[:60] for p in boundary_papers[:3]]
    
    templates = [
        {
            'title': f'Bridging {topics[0].split(":")[0] if ":" in topics[0] else topics[0][:40]} with Neural Approaches',
            'description': 'Explore hybrid methods that combine the strengths of existing techniques with modern neural architectures to address open challenges in this sparse research area.'
        },
        {
            'title': f'Cross-domain Transfer for {topics[-1].split()[-1] if topics else "Research"} Problems',
            'description': 'Investigate transfer learning strategies to leverage knowledge from adjacent well-studied domains and apply them to underexplored regions of this research space.'
        },
        {
            'title': 'Unified Framework for Multi-modal Integration in This Research Gap',
            'description': 'Propose a unified architecture that simultaneously handles multiple input modalities, addressing the gaps identified between the boundary papers in this sparse cluster region.'
        }
    ]
    
    for t in templates[:len(boundary_papers) + 1]:
        ideas.append(t)
    
    return ideas[:3]

def find_gap_for_topic(papers_db, embeddings, t_sne_coords, topic_keyword, api_key, use_groq=False):
    """
    Main entry point: find a gap for a specific research topic.
    """
    # Filter papers by keyword
    topic_papers = [
        (i, p) for i, p in enumerate(papers_db)
        if topic_keyword.lower() in (p.get('title', '') + ' ' + p.get('abstract', '')).lower()
    ]
    
    if not topic_papers:
        return {
            'error': f'No papers found for topic: {topic_keyword}',
            'ideas': []
        }
    
    # Find sparse regions
    if len(t_sne_coords) > 0:
        sparse_regions = find_sparse_regions(embeddings, t_sne_coords, num_regions=3)
    else:
        sparse_regions = []
    
    # Generate ideas for top sparse region
    if sparse_regions:
        ideas_result = generate_idea_suggestions(
            sparse_regions[0], papers_db, embeddings, api_key, use_groq
        )
        return {
            'topic': topic_keyword,
            'sparse_regions': [{
                'x': float(r['x']),
                'y': float(r['y']),
                'paper_indices': [int(i) for i in r['paper_indices']],
                'sparsity_score': float(r['sparsity_score'])
            } for r in sparse_regions],
            'ideas': ideas_result['ideas'],
            'novelty_scores': [float(s) for s in ideas_result['novelty_scores']],
            'supporting_papers': ideas_result.get('supporting_papers', [])
        }
    
    return {
        'topic': topic_keyword,
        'error': 'Unable to identify sparse research regions',
        'ideas': []
    }
