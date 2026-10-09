"""
Author and Venue Network — Part 2 Feature #5
Force-directed graph of co-authors and topic neighbors
Integrates with Semantic Scholar API for citations
"""

import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

@dataclass
class Author:
    """Author node in network"""
    author_id: str
    name: str
    paper_count: int
    h_index: int  # citation impact
    research_areas: List[str]
    affiliation: str
    papers: List[str]  # Paper IDs

@dataclass
class Venue:
    """Venue node in network"""
    venue_id: str
    name: str
    type: str  # "conference", "journal", "workshop"
    papers: List[str]
    impact_factor: float
    avg_citations: float

@dataclass
class NetworkEdge:
    """Edge in the graph"""
    source_id: str
    target_id: str
    source_type: str  # "author", "venue", "topic"
    target_type: str
    weight: float  # Collaboration strength
    papers: List[str]  # Shared papers
    label: str  # "co-authored N papers", "published N papers"

class AuthorVenueNetwork:
    """Build and analyze co-author and venue networks"""
    
    def __init__(self, papers: List[Dict], semantic_scholar_client=None):
        self.papers = papers
        self.semantic_scholar = semantic_scholar_client
        self.author_index = {}  # author_id -> Author
        self.venue_index = {}   # venue_id -> Venue
        self.edges = []
        
        # Build network
        self._build_author_index()
        self._build_venue_index()
        self._build_edges()
    
    def _build_author_index(self):
        """Index all authors"""
        
        author_papers = defaultdict(list)
        author_info = {}
        
        for paper in self.papers:
            authors = paper.get("authors", [])
            for author in authors:
                author_id = author.get("id") or author.get("name").lower().replace(" ", "_")
                author_papers[author_id].append(paper["id"])
                
                if author_id not in author_info:
                    author_info[author_id] = {
                        "name": author.get("name", "Unknown"),
                        "affiliation": author.get("affiliation", ""),
                        "research_areas": author.get("research_areas", [])
                    }
        
        # Create Author objects
        for author_id, paper_ids in author_papers.items():
            info = author_info[author_id]
            author = Author(
                author_id=author_id,
                name=info["name"],
                paper_count=len(paper_ids),
                h_index=self._estimate_h_index(paper_ids),
                research_areas=info["research_areas"],
                affiliation=info["affiliation"],
                papers=paper_ids
            )
            self.author_index[author_id] = author
    
    def _build_venue_index(self):
        """Index all venues (conferences, journals)"""
        
        venue_papers = defaultdict(list)
        venue_info = {}
        
        for paper in self.papers:
            venue = paper.get("venue", {})
            if isinstance(venue, str):
                venue_id = venue.lower().replace(" ", "_")
                venue_name = venue
            else:
                venue_id = venue.get("id", "unknown")
                venue_name = venue.get("name", "Unknown")
            
            venue_papers[venue_id].append(paper["id"])
            
            if venue_id not in venue_info:
                venue_info[venue_id] = {
                    "name": venue_name,
                    "type": venue.get("type", "conference"),
                    "impact_factor": venue.get("impact_factor", 0.0)
                }
        
        # Create Venue objects
        for venue_id, paper_ids in venue_papers.items():
            info = venue_info[venue_id]
            avg_citations = np.mean([
                self.papers[i].get("citation_count", 0)
                for i, p in enumerate(self.papers) if p["id"] in paper_ids
            ])
            
            venue = Venue(
                venue_id=venue_id,
                name=info["name"],
                type=info["type"],
                papers=paper_ids,
                impact_factor=info.get("impact_factor", 0.0),
                avg_citations=avg_citations
            )
            self.venue_index[venue_id] = venue
    
    def _build_edges(self):
        """Build edges between nodes"""
        
        # Author-to-author edges (co-authorship)
        for paper in self.papers:
            authors = [a.get("id") or a.get("name").lower().replace(" ", "_") 
                      for a in paper.get("authors", [])]
            
            # Create edges between all author pairs
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    # Find existing edge or create new one
                    existing_edge = next(
                        (e for e in self.edges 
                         if (e.source_id == author1 and e.target_id == author2) or
                            (e.source_id == author2 and e.target_id == author1)),
                        None
                    )
                    
                    if existing_edge:
                        existing_edge.papers.append(paper["id"])
                        existing_edge.weight += 1.0
                    else:
                        edge = NetworkEdge(
                            source_id=author1,
                            target_id=author2,
                            source_type="author",
                            target_type="author",
                            weight=1.0,
                            papers=[paper["id"]],
                            label=f"Co-authored {len([paper['id']])} paper(s)"
                        )
                        self.edges.append(edge)
        
        # Venue-to-paper edges (publication)
        for paper in self.papers:
            venue = paper.get("venue", {})
            venue_id = venue.lower().replace(" ", "_") if isinstance(venue, str) else venue.get("id", "unknown")
            
            edge = NetworkEdge(
                source_id=venue_id,
                target_id=paper["id"],
                source_type="venue",
                target_type="paper",
                weight=1.0,
                papers=[paper["id"]],
                label="Published"
            )
            self.edges.append(edge)
    
    def get_network_graph(
        self,
        node_types: List[str] = ["author", "venue"],
        top_authors: int = 20,
        min_collaboration: int = 1
    ) -> Dict:
        """Export network as nodes and edges for visualization"""
        
        nodes = []
        edges = []
        
        # Add author nodes
        if "author" in node_types:
            # Sort by paper count and take top N
            sorted_authors = sorted(
                self.author_index.values(),
                key=lambda a: a.paper_count,
                reverse=True
            )[:top_authors]
            
            for author in sorted_authors:
                nodes.append({
                    "id": author.author_id,
                    "type": "author",
                    "label": author.name,
                    "size": min(50, author.paper_count * 2),  # Node size by paper count
                    "color": "#FF6B6B",
                    "metadata": {
                        "paper_count": author.paper_count,
                        "h_index": author.h_index,
                        "affiliation": author.affiliation,
                        "research_areas": author.research_areas
                    }
                })
        
        # Add venue nodes
        if "venue" in node_types:
            for venue in self.venue_index.values():
                nodes.append({
                    "id": venue.venue_id,
                    "type": "venue",
                    "label": venue.name,
                    "size": min(40, len(venue.papers)),
                    "color": "#4ECDC4",
                    "metadata": {
                        "type": venue.type,
                        "paper_count": len(venue.papers),
                        "avg_citations": venue.avg_citations,
                        "impact_factor": venue.impact_factor
                    }
                })
        
        # Add edges (only between included nodes)
        included_nodes = {n["id"] for n in nodes}
        
        for edge in self.edges:
            if edge.source_id not in included_nodes or edge.target_id not in included_nodes:
                continue
            
            if edge.weight < min_collaboration:
                continue
            
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "weight": edge.weight,
                "label": edge.label,
                "type": f"{edge.source_type}-to-{edge.target_type}"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_authors": len(self.author_index),
                "total_venues": len(self.venue_index),
                "total_papers": len(self.papers),
                "visualization_type": "force-directed graph"
            }
        }
    
    def get_author_profile(self, author_id: str) -> Dict:
        """Get detailed profile for an author"""
        
        if author_id not in self.author_index:
            return None
        
        author = self.author_index[author_id]
        
        # Get co-authors
        co_authors = defaultdict(int)
        for paper_id in author.papers:
            paper = next((p for p in self.papers if p["id"] == paper_id), None)
            if paper:
                for co in paper.get("authors", []):
                    co_id = co.get("id") or co.get("name").lower().replace(" ", "_")
                    if co_id != author_id:
                        co_authors[co_id] += 1
        
        # Get venues where author published
        venues = defaultdict(int)
        for paper_id in author.papers:
            paper = next((p for p in self.papers if p["id"] == paper_id), None)
            if paper:
                venue = paper.get("venue", {})
                venue_id = venue.lower().replace(" ", "_") if isinstance(venue, str) else venue.get("id", "unknown")
                venues[venue_id] += 1
        
        return {
            "id": author_id,
            "name": author.name,
            "affiliation": author.affiliation,
            "paper_count": author.paper_count,
            "h_index": author.h_index,
            "research_areas": author.research_areas,
            "co_authors": [
                {
                    "id": co_id,
                    "name": self.author_index[co_id].name,
                    "collaboration_count": count
                }
                for co_id, count in sorted(co_authors.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "venues": [
                {
                    "id": v_id,
                    "name": self.venue_index[v_id].name,
                    "publication_count": count
                }
                for v_id, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            "papers": author.papers[:10]  # First 10 papers
        }
    
    def find_collaboration_paths(
        self,
        author1_id: str,
        author2_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """Find shortest paths between two authors (degrees of separation)"""
        
        if author1_id not in self.author_index or author2_id not in self.author_index:
            return []
        
        # BFS to find paths
        visited = {author1_id}
        queue = [(author1_id, [author1_id])]
        paths = []
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            # Find co-authors
            current_author = self.author_index.get(current)
            if not current_author:
                continue
            
            co_authors = set()
            for paper_id in current_author.papers:
                paper = next((p for p in self.papers if p["id"] == paper_id), None)
                if paper:
                    for author in paper.get("authors", []):
                        co_id = author.get("id") or author.get("name").lower().replace(" ", "_")
                        if co_id != current:
                            co_authors.add(co_id)
            
            for co_author in co_authors:
                if co_author == author2_id:
                    paths.append(path + [co_author])
                elif co_author not in visited and len(path) < max_depth:
                    visited.add(co_author)
                    queue.append((co_author, path + [co_author]))
        
        return paths
    
    def get_trending_collaborations(self, years: int = 2) -> List[Dict]:
        """Get trending author pairs (recent collaborations)"""
        
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=365*years)
        
        collaboration_scores = defaultdict(int)
        
        for paper in self.papers:
            paper_year = paper.get("year", 0)
            if paper_year < cutoff_date.year:
                continue
            
            authors = [a.get("id") or a.get("name").lower().replace(" ", "_") 
                      for a in paper.get("authors", [])]
            
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    key = tuple(sorted([a1, a2]))
                    collaboration_scores[key] += 1
        
        # Convert to list and sort
        trending = [
            {
                "author1_id": a1,
                "author2_id": a2,
                "author1_name": self.author_index[a1].name if a1 in self.author_index else a1,
                "author2_name": self.author_index[a2].name if a2 in self.author_index else a2,
                "collaboration_count": score
            }
            for (a1, a2), score in sorted(collaboration_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return trending[:20]
    
    def _estimate_h_index(self, paper_ids: List[str]) -> int:
        """Estimate h-index from papers"""
        
        citations = [
            next((p.get("citation_count", 0) for p in self.papers if p["id"] == pid), 0)
            for pid in paper_ids
        ]
        
        citations.sort(reverse=True)
        
        h_index = 0
        for i, citations_count in enumerate(citations):
            if citations_count >= i + 1:
                h_index = i + 1
            else:
                break
        
        return h_index
