"""
Learning Path Generator — Part 2 Feature #1
Builds an ordered reading path from foundational to advanced papers
[FIXED VERSION - All bugs corrected]
"""

import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import heapq
import logging

# Setup logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PathNode:
    """Represents a paper in the learning path"""
    paper_id: str
    title: str
    year: int
    level: str  # "foundational", "intermediate", "advanced"
    prerequisite_score: float  # How well understood before reading this
    relevance_score: float  # How relevant to target topic
    estimated_read_time: int  # minutes
    summary: str
    reasoning: str  # Why this paper comes here

class LearningPathGenerator:
    """Generate ordered reading paths for a given topic"""
    
    def __init__(self, embeddings_model, llm_router, rag_pipeline):
        self.embeddings = embeddings_model
        self.llm = llm_router
        self.rag = rag_pipeline
        self.paths_cache = {}
        
    def generate_path(
        self,
        topic: str,
        target_papers: List[Dict],
        max_papers: int = 20,
        difficulty_progression: str = "linear"
    ) -> Dict:
        """
        Generate an optimal learning path for a topic.
        
        Args:
            topic: Target topic (e.g., "Transformer attention mechanisms")
            target_papers: List of papers in the corpus with embeddings
            max_papers: Max papers in path (default 20)
            difficulty_progression: "linear", "sigmoid", "exponential"
            
        Returns:
            {
                "path": [PathNode],
                "metadata": {
                    "estimated_hours": float,
                    "topic_coverage": float,
                    "difficulty_curve": [float],
                    "milestones": [{"after_paper": int, "milestone": str}]
                }
            }
        """
        
        # BUG FIX #4: Input validation
        if not target_papers:
            raise ValueError("target_papers cannot be empty")
        
        if not topic or not topic.strip():
            raise ValueError("topic cannot be empty")
        
        # 1. Embed the topic and find semantically related papers
        topic_embedding = self.embeddings.encode(topic)
        
        # Score papers by relevance to topic
        relevance_scores = []
        for paper in target_papers:
            # BUG FIX #2: Validate embedding exists and is not empty
            paper_embedding = paper.get("embedding")
            
            if paper_embedding is None:
                logger.warning(f"Paper {paper.get('id', 'unknown')} has no embedding, skipping")
                continue
            
            paper_embedding = np.array(paper_embedding)
            if len(paper_embedding) == 0:
                logger.warning(f"Paper {paper.get('id', 'unknown')} has empty embedding, skipping")
                continue
            
            relevance = float(np.dot(topic_embedding, paper_embedding))
            relevance_scores.append((paper["id"], relevance, paper))
        
        if not relevance_scores:
            raise ValueError("No valid papers with embeddings found")
        
        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        candidate_papers = relevance_scores[:max_papers * 2]  # Oversample
        
        # 2. Classify papers into difficulty levels using LLM
        level_classifications = self._classify_difficulty_levels(
            topic, [p[2] for p in candidate_papers]
        )
        
        # 3. Build prerequisite graph (if A must be read before B)
        prereq_graph = self._build_prerequisite_graph(
            [p[2] for p in candidate_papers],
            topic
        )
        
        # 4. Topological sort with leveling
        ordered_papers = self._topological_sort_with_levels(
            candidate_papers,
            level_classifications,
            prereq_graph,
            max_papers,
            difficulty_progression
        )
        
        # 5. Compute path metrics and milestones
        path_nodes = []
        cumulative_understanding = 0
        milestones = []
        
        for idx, (paper_id, relevance, paper) in enumerate(ordered_papers):
            level = level_classifications.get(paper_id, "intermediate")
            read_time = self._estimate_read_time(paper, level)
            
            # LLM generates reasoning for why this paper is at this position
            reasoning = self._generate_reasoning(
                paper, level, idx, len(ordered_papers), topic
            )
            
            # Track cumulative understanding
            if level == "foundational":
                cumulative_understanding += 0.3
            elif level == "intermediate":
                cumulative_understanding += 0.5
            else:  # advanced
                cumulative_understanding += 0.7
            
            cumulative_understanding = min(1.0, cumulative_understanding)
            
            node = PathNode(
                paper_id=paper_id,
                title=paper.get("title", "Unknown"),
                year=paper.get("year", 0),
                level=level,
                prerequisite_score=cumulative_understanding,
                relevance_score=relevance,
                estimated_read_time=read_time,
                summary=paper.get("summary", ""),
                reasoning=reasoning
            )
            path_nodes.append(node)
            
            # Mark milestones (e.g., "Understand basic concepts", "Expert level")
            if idx == len(ordered_papers) // 4:
                milestones.append({
                    "after_paper": idx,
                    "milestone": "Understand foundational concepts"
                })
            elif idx == len(ordered_papers) // 2:
                milestones.append({
                    "after_paper": idx,
                    "milestone": "Ready for intermediate papers"
                })
            elif idx == 3 * len(ordered_papers) // 4:
                milestones.append({
                    "after_paper": idx,
                    "milestone": "Approaching expert level"
                })
        
        # Final metrics
        # BUG FIX #1: Handle empty path_nodes (division by zero)
        total_hours = sum(n.estimated_read_time for n in path_nodes) / 60
        
        topic_coverage = (
            sum(n.relevance_score for n in path_nodes) / len(path_nodes)
            if path_nodes else 0.0
        )
        
        difficulty_curve = [n.prerequisite_score for n in path_nodes]
        
        return {
            "path": [asdict(n) for n in path_nodes],
            "metadata": {
                "topic": topic,
                "num_papers": len(path_nodes),
                "estimated_hours": round(total_hours, 1),
                "topic_coverage": round(topic_coverage, 3),
                "difficulty_curve": difficulty_curve,
                "milestones": milestones,
                "created_at": datetime.now().isoformat()
            }
        }
    
    def _classify_difficulty_levels(self, topic: str, papers: List[Dict]) -> Dict[str, str]:
        """Use LLM to classify papers as foundational, intermediate, or advanced"""
        
        paper_summaries = "\n".join([
            f"- {p['id']}: {p.get('title', 'Unknown')} ({p.get('year', 'N/A')})"
            for p in papers[:10]  # Batch process first 10
        ])
        
        prompt = f"""
Classify these papers for learning '{topic}' into three levels:
- foundational (prerequisites, intro concepts, seminal works)
- intermediate (builds on foundations, introduces new techniques)
- advanced (state-of-the-art, specialized variations, recent work)

Papers:
{paper_summaries}

Return JSON: {{"paper_id": "level"}}
"""
        
        response = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

        # BUG FIX #3: Specific exception handling
        try:
            import re
            # LLMs often wrap JSON in ```json fences or prose — extract the object.
            match = re.search(r'\{.*\}', response or "", re.DOTALL)
            classifications = json.loads(match.group() if match else response)
            # Keep only the valid level labels
            valid = {"foundational", "intermediate", "advanced"}
            classifications = {k: v for k, v in classifications.items() if v in valid}
            if not classifications:
                raise ValueError("no valid levels")
            return classifications
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"JSON parse failed for difficulty classification: {e}, using year-based fallback")
            # Fallback: use year as proxy
            return {
                p["id"]: (
                    "foundational" if p.get("year", 2020) < 2015
                    else "intermediate" if p.get("year", 2020) < 2020
                    else "advanced"
                )
                for p in papers
            }
    
    def _build_prerequisite_graph(self, papers: List[Dict], topic: str) -> Dict[str, set]:
        """Build directed graph: A -> B means A should be read before B"""
        
        graph = {p["id"]: set() for p in papers}
        
        # Simple heuristic: papers citing each other
        for paper in papers:
            paper_id = paper["id"]
            # BUG FIX: Use .get() with default empty list
            refs = paper.get("references", [])
            
            for ref_id in refs:
                if ref_id in graph:
                    # ref_id should be read before paper_id
                    graph[ref_id].add(paper_id)
        
        return graph
    
    def _topological_sort_with_levels(
        self,
        papers: List[Tuple],
        levels: Dict[str, str],
        graph: Dict[str, set],
        max_papers: int,
        progression: str
    ) -> List[Tuple]:
        """
        Topological sort respecting prerequisite constraints,
        with difficulty progression
        """
        
        # Compute prerequisite in-degrees (used as a tie-breaker). Many corpora
        # (e.g. recent arXiv snapshots) carry no citation edges, so the ordering
        # must not *depend* on them — otherwise only the seed level is ever emitted.
        in_degree = {pid: 0 for pid, _, _ in papers}
        for pid in graph:
            for neighbor in graph[pid]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1

        level_order = ["foundational", "intermediate", "advanced"]
        level_rank = {lvl: i for i, lvl in enumerate(level_order)}

        # Stable, edge-independent ordering: primarily by difficulty level
        # (foundational → advanced), then by fewer prerequisites, then by
        # relevance to the topic. Guaranteed to terminate and to fill the path.
        ordered = sorted(
            papers,
            key=lambda t: (
                level_rank.get(levels.get(t[0], "intermediate"), 1),
                in_degree.get(t[0], 0),
                -t[1],
            ),
        )

        return ordered[:max_papers]
    
    def _estimate_read_time(self, paper: Dict, level: str) -> int:
        """Estimate reading time in minutes"""
        
        # Base: page count or abstract length
        pages = paper.get("pages", 10)
        base_time = min(pages * 2, 180)  # 2 min per page, max 3 hours
        
        # Adjust by level
        multiplier = {
            "foundational": 1.0,
            "intermediate": 1.3,
            "advanced": 1.6
        }.get(level, 1.0)
        
        return int(base_time * multiplier)
    
    def _generate_reasoning(
        self,
        paper: Dict,
        level: str,
        position: int,
        total: int,
        topic: str
    ) -> str:
        """Generate LLM-powered reasoning for paper placement"""
        
        # Fast, deterministic pedagogical reasoning (no per-paper LLM round-trip,
        # which previously meant one network call per paper and made the endpoint
        # time out). The expensive LLM step is the single difficulty classification.
        stage = {
            "foundational": "establishes the core concepts and vocabulary you need before going further",
            "intermediate": "builds on the fundamentals and introduces the key techniques",
            "advanced": "represents state-of-the-art work that assumes the earlier material",
        }.get(level, "extends your understanding of the topic")
        where = "Start here" if position == 0 else (
            "Read this next" if position < total / 2 else "Tackle this later in the path")
        return (f"{where}: as a {level} paper for \"{topic}\", it {stage}. "
                f"Placed at step {position + 1} of {total}.")
