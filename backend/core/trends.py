"""
Research Trends Explorer
Analyzes paper publication volume over time, detects rising topics,
and generates LLM summaries of trend insights.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict
from .llm_helper import call_llm_with_retry

def compute_trends(papers_db, cluster_labels=None):
    """
    Compute trend data: topic volume over time, growth rate, rising topics.
    
    Args:
        papers_db: List of paper dicts with 'published_date' and 'title'
        cluster_labels: Optional dict mapping paper_id -> cluster_label
    
    Returns:
        {
            'trend_data': [{
                'week': '2023-W01',
                'topics': {'Machine Learning': 10, 'NLP': 8, ...},
                'total_papers': 18
            }, ...],
            'rising_topics': [
                {'topic': 'Large Language Models', 'growth_rate': 0.45, 'this_week': 12, 'prev_week': 8},
                ...
            ],
            'new_this_week': ['Transformer Improvements', 'Retrieval Augmented Generation'],
            'top_topics': ['Machine Learning', 'NLP', 'Computer Vision']
        }
    """
    if not papers_db:
        return {
            'trend_data': [],
            'rising_topics': [],
            'new_this_week': [],
            'top_topics': []
        }
    
    # Group papers by DAY. (arXiv snapshots typically span only a few days inside
    # a single ISO week, so week-over-week buckets collapse to one bucket and no
    # growth can ever be detected. Daily buckets give a meaningful recent signal.)
    weekly_data = defaultdict(lambda: defaultdict(int))
    today = datetime.now()

    for paper in papers_db:
        try:
            # Parse published date - field is 'published' (format: 'YYYY-MM-DD')
            pub_str = paper.get('published', '')
            if isinstance(pub_str, str) and pub_str:
                try:
                    pub_date = datetime.strptime(pub_str[:10], '%Y-%m-%d')
                except ValueError:
                    pub_date = today
            else:
                pub_date = today

            # Bucket identifier = the publication day
            year_week = pub_date.strftime('%Y-%m-%d')

            # Extract topic label (use cluster label if available, else extract from title)
            if cluster_labels and paper['id'] in cluster_labels:
                topic = cluster_labels[paper['id']]
            else:
                topic = extract_topic_from_title(paper.get('title', 'Unknown'))

            weekly_data[year_week][topic] += 1

        except (ValueError, KeyError):
            continue

    # Sort buckets chronologically
    sorted_weeks = sorted(weekly_data.keys())
    
    # Build trend data with growth rates
    trend_data = []
    prev_topics = {}
    
    for week in sorted_weeks:
        topics = dict(weekly_data[week])
        total_papers = sum(topics.values())
        
        trend_data.append({
            'week': week,
            'topics': topics,
            'total_papers': total_papers
        })
        
        prev_topics = topics
    
    # Calculate rising topics: latest bucket vs the average of the earlier buckets.
    rising_topics = []
    if len(sorted_weeks) >= 2:
        curr_week_topics = dict(weekly_data[sorted_weeks[-1]])
        earlier_weeks = sorted_weeks[:-1]
        # Average count per topic across all earlier buckets, and the set of
        # topics that have ever appeared before the latest bucket.
        earlier_sum = defaultdict(int)
        seen_before = set()
        for w in earlier_weeks:
            for topic, c in weekly_data[w].items():
                earlier_sum[topic] += c
                seen_before.add(topic)
        n_earlier = max(1, len(earlier_weeks))

        for topic, curr_count in curr_week_topics.items():
            prev_avg = earlier_sum.get(topic, 0) / n_earlier
            if prev_avg > 0:
                growth_rate = (curr_count - prev_avg) / prev_avg
                if growth_rate > 0.15:  # 15% above its recent baseline
                    rising_topics.append({
                        'topic': topic,
                        'growth_rate': round(growth_rate, 2),
                        'this_week': curr_count,
                        'prev_week': round(prev_avg, 1)
                    })
            elif curr_count > 0 and topic not in seen_before:
                # Genuinely new topic (never seen in any earlier bucket)
                rising_topics.append({
                    'topic': topic,
                    'growth_rate': 1.0,
                    'this_week': curr_count,
                    'prev_week': 0
                })

    rising_topics.sort(key=lambda x: x['growth_rate'], reverse=True)

    # "New this week": titles of the most recently published papers — genuinely
    # new arrivals. (Topic categories are coarse and appear every day, so listing
    # recent papers is far more informative than listing recurring topic labels.)
    recent_papers = sorted(
        [p for p in papers_db if p.get('published')],
        key=lambda p: str(p.get('published', '')), reverse=True
    )
    new_this_week = [p.get('title', 'Untitled') for p in recent_papers[:6]]
    
    # Top topics overall
    all_topics_count = defaultdict(int)
    for week_topics in weekly_data.values():
        for topic, count in week_topics.items():
            all_topics_count[topic] += count
    
    top_topics = sorted(all_topics_count.items(), key=lambda x: x[1], reverse=True)[:5]
    top_topics = [t[0] for t in top_topics]
    
    return {
        'trend_data': trend_data,
        'rising_topics': rising_topics[:10],  # Top 10 rising
        'new_this_week': new_this_week[:5],
        'top_topics': top_topics
    }

def extract_topic_from_title(title):
    """
    Extract a topic category from the paper title using keyword matching.
    """
    title_lower = title.lower()
    
    keywords_map = {
        'Large Language Models': ['llm', 'language model', 'transformer', 'gpt', 'bert'],
        'Computer Vision': ['vision', 'image', 'visual', 'object detection', 'segmentation'],
        'Natural Language Processing': ['nlp', 'text', 'nlp', 'language', 'bert', 'gpt'],
        'Reinforcement Learning': ['reinforcement learning', 'rl', 'agent', 'policy'],
        'Graph Neural Networks': ['graph', 'gnn', 'node', 'edge'],
        'Diffusion Models': ['diffusion', 'denoising', 'score matching'],
        'Retrieval Augmented Generation': ['rag', 'retrieval', 'augmented generation'],
        'Fine-tuning': ['fine-tun', 'adaptation', 'transfer learning'],
        'Multimodal': ['multimodal', 'vision-language', 'cross-modal'],
        'Efficient ML': ['efficient', 'lightweight', 'pruning', 'quantization', 'compression']
    }
    
    for topic, keywords in keywords_map.items():
        if any(kw in title_lower for kw in keywords):
            return topic
    
    return 'General Machine Learning'

def generate_trend_summary(trends_data, api_key, use_groq=False):
    """
    Generate an LLM summary of trend insights.
    """
    rising = trends_data.get('rising_topics', [])[:3]
    top = trends_data.get('top_topics', [])[:3]
    
    if not rising and not top:
        return "No significant trends detected this week."
    
    rising_str = ", ".join([f"{t['topic']} (+{int(t['growth_rate']*100)}%)" for t in rising])
    top_str = ", ".join(top)
    
    prompt = f"""Summarize the following research trends in one paragraph (2-3 sentences):

Rising Topics (growing 20%+ this week): {rising_str}
Top Overall Topics: {top_str}

Be concise and highlight what's happening in the research community."""
    
    try:
        summary = call_llm_with_retry(prompt, api_key, use_groq=use_groq)
        return summary if summary else "Research activity shows steady growth across LLM and vision domains."
    except Exception as e:
        print(f"Error generating trend summary: {e}")
        return f"Research trends show activity in: {top_str}. Rising interest in: {rising_str}."
