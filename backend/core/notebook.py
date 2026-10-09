"""
Notes Notebook (Second Brain) — Part 2 Feature #2
Attach notes to papers, semantic search, auto-linking to related papers
[FIXED VERSION - All bugs corrected]
"""

import json
import uuid
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (range -1 to 1)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    # Guard against mismatched embedding dimensions (e.g. TF-IDF vs SBERT)
    if a.shape != b.shape:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

@dataclass
class Note:
    """A note attached to a paper"""
    note_id: str
    paper_id: str
    title: str
    content: str  # Markdown
    tags: List[str]
    embedding: List[float]  # Embedded using sentence-transformers
    created_at: str
    updated_at: str
    is_public: bool = False

@dataclass
class NotebookLink:
    """Auto-discovered link between notes"""
    from_note_id: str
    to_note_id: str
    to_paper_id: str
    similarity_score: float
    link_reason: str  # Why they're related

class NotebookManager:
    """Semantic notebook for research notes"""
    
    def __init__(self, embeddings_model, llm_router, storage_path: str = "data/notebook.jsonl"):
        self.embeddings = embeddings_model
        self.llm = llm_router
        self.storage_path = storage_path
        # BUG FIX #6: Create directory structure
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.notes = self._load_notes()
        self.paper_index = {}  # paper_id -> [note_ids]
    
    def create_note(
        self,
        paper_id: str,
        title: str,
        content: str,
        tags: List[str] = None,
        is_public: bool = False
    ) -> Note:
        """Create a new note attached to a paper"""
        
        note_id = str(uuid.uuid4())[:8]
        embedding = self.embeddings.encode(title + " " + content)
        
        note = Note(
            note_id=note_id,
            paper_id=paper_id,
            title=title,
            content=content,
            tags=tags or [],
            embedding=embedding.tolist(),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            is_public=is_public
        )
        
        self.notes[note_id] = note
        
        # Index
        if paper_id not in self.paper_index:
            self.paper_index[paper_id] = []
        self.paper_index[paper_id].append(note_id)
        
        # Find and store auto-links
        self._find_and_store_links(note_id)
        
        # Persist
        self._save_notes()
        
        return note
    
    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None
    ) -> Note:
        """Update an existing note"""
        
        if note_id not in self.notes:
            raise ValueError(f"Note {note_id} not found")
        
        note = self.notes[note_id]
        
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
            # Re-embed when content changes
            note.embedding = self.embeddings.encode(
                note.title + " " + note.content
            ).tolist()
        if tags is not None:
            note.tags = tags
        if is_public is not None:
            note.is_public = is_public
        
        note.updated_at = datetime.now().isoformat()
        
        # Re-find links
        self._find_and_store_links(note_id)
        self._save_notes()
        
        return note
    
    def delete_note(self, note_id: str) -> bool:
        """Delete a note"""
        
        if note_id not in self.notes:
            return False
        
        note = self.notes[note_id]
        del self.notes[note_id]
        
        # Remove from paper index
        if note.paper_id in self.paper_index:
            self.paper_index[note.paper_id].remove(note_id)
        
        # Remove links involving this note
        # (simplified: just re-build on next find)
        
        self._save_notes()
        return True
    
    def get_notes_for_paper(self, paper_id: str) -> List[Dict]:
        """Get all notes for a paper"""
        
        note_ids = self.paper_index.get(paper_id, [])
        notes = [asdict(self.notes[nid]) for nid in note_ids if nid in self.notes]
        
        return sorted(notes, key=lambda n: n["updated_at"], reverse=True)
    
    def search_notes(
        self,
        query: str,
        top_k: int = 10,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """Semantic search across all notes"""
        
        query_embedding = self.embeddings.encode(query)
        scores = []
        
        for note_id, note in self.notes.items():
            # BUG FIX #7: Clarify tag filter logic with comments
            # Include note if it has ANY of the filter tags (OR logic)
            if tag_filter:
                if not any(t in note.tags for t in tag_filter):
                    # For AND logic instead, use: if not all(t in note.tags for t in tag_filter)
                    continue
            
            # Cosine similarity (works correctly regardless of embedding norm)
            similarity = _cosine_similarity(query_embedding, note.embedding)
            scores.append((note_id, similarity, note))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                **asdict(note),
                "search_score": score
            }
            for _, score, note in scores[:top_k]
        ]
    
    def _find_and_store_links(self, note_id: str) -> List[NotebookLink]:
        """Find semantically related notes and return link objects"""
        
        if note_id not in self.notes:
            return []
        
        note = self.notes[note_id]
        note_embedding = np.array(note.embedding)
        
        links = []
        
        # Find similar notes
        similarities = []
        for other_id, other_note in self.notes.items():
            if other_id == note_id:
                continue
            
            sim = _cosine_similarity(note.embedding, other_note.embedding)
            similarities.append((other_id, sim, other_note))
        
        # Get top 5 most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        for other_id, sim_score, other_note in similarities[:5]:
            if sim_score < 0.5:  # Threshold
                break
            
            # Generate link reason
            reason = self._generate_link_reason(note, other_note, sim_score)
            
            link = NotebookLink(
                from_note_id=note_id,
                to_note_id=other_id,
                to_paper_id=other_note.paper_id,
                similarity_score=sim_score,
                link_reason=reason
            )
            links.append(link)
        
        return links
    
    def _generate_link_reason(self, note1: Note, note2: Note, sim_score: float) -> str:
        """Generate LLM-powered reason for why two notes are related"""
        
        prompt = f"""
Given these two research notes, explain in one sentence why they're related:

Note 1 (Paper {note1.paper_id}): {note1.title}
Content: {note1.content[:200]}

Note 2 (Paper {note2.paper_id}): {note2.title}
Content: {note2.content[:200]}

Similarity: {sim_score:.2f}

Why are they related?
"""
        
        return self.llm.generate(prompt, temperature=0.5, max_tokens=100)
    
    def get_related_papers(self, note_id: str, top_k: int = 5) -> Dict:
        """Get papers related to this note's concepts
        
        BUG FIX #5: Return type is Dict, not List[Dict]
        """
        
        if note_id not in self.notes:
            return {
                "note_id": note_id,
                "note_title": "Not Found",
                "related_papers": [],
                "methodology": "Semantic similarity + tag-based filtering"
            }
        
        note = self.notes[note_id]
        
        # This would integrate with your corpus search
        # For now, return structure
        return {
            "note_id": note_id,
            "note_title": note.title,
            "related_papers": [],  # Would be filled by calling search on corpus
            "methodology": "Semantic similarity + tag-based filtering"
        }
    
    def get_notebook_graph(self) -> Dict:
        """Export the entire notebook as a knowledge graph"""
        
        nodes = [
            {
                "id": note_id,
                "type": "note",
                "label": note.title,
                "paper_id": note.paper_id,
                "tags": note.tags,
                "created_at": note.created_at
            }
            for note_id, note in self.notes.items()
        ]
        
        edges = []
        
        # Find all similarity links
        for note_id, note in self.notes.items():
            note_embedding = np.array(note.embedding)
            
            for other_id, other_note in self.notes.items():
                if note_id >= other_id:  # Avoid duplicates
                    continue
                
                sim = _cosine_similarity(note.embedding, other_note.embedding)
                
                if sim > 0.5:
                    edges.append({
                        "from": note_id,
                        "to": other_id,
                        "weight": sim,
                        "type": "similar"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "total_notes": len(self.notes),
                "total_papers": len(self.paper_index),
                "total_links": len(edges)
            }
        }
    
    def export_notebook(self, format: str = "markdown") -> str:
        """Export notebook in various formats"""
        
        if format == "markdown":
            lines = ["# Research Notebook\n"]
            
            # Group by paper
            for paper_id, note_ids in self.paper_index.items():
                lines.append(f"\n## Paper: {paper_id}\n")
                
                for note_id in note_ids:
                    if note_id in self.notes:
                        note = self.notes[note_id]
                        lines.append(f"### {note.title}\n")
                        lines.append(note.content + "\n")
                        
                        if note.tags:
                            lines.append(f"**Tags:** {', '.join(note.tags)}\n")
                        
                        lines.append(f"*Updated: {note.updated_at}*\n")
            
            return "\n".join(lines)
        
        elif format == "json":
            return json.dumps({
                "notes": {
                    nid: asdict(note) for nid, note in self.notes.items()
                },
                "created_at": datetime.now().isoformat()
            }, indent=2, default=str)
        
        return ""
    
    def _load_notes(self) -> Dict[str, Note]:
        """Load notes from storage
        
        BUG FIX #6: Handle missing files and parse errors gracefully
        """
        
        # Check if file exists
        if not os.path.exists(self.storage_path):
            logger.info(f"Note storage file does not exist yet: {self.storage_path}")
            return {}
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                notes = {}
                for nid, note_data in data.items():
                    try:
                        notes[nid] = Note(**note_data)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to parse note {nid}: {e}")
                        continue
                return notes
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading notes: {e}")
            return {}
    
    def _save_notes(self):
        """Persist notes to storage
        
        BUG FIX #6: Ensure directory exists before writing
        """
        
        # Ensure directory exists
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                nid: asdict(note)
                for nid, note in self.notes.items()
            }
            # Atomic write: write to a temp file then rename to avoid
            # Windows/OneDrive file-lock issues (OSError [Errno 22])
            tmp_path = self.storage_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, self.storage_path)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save notes: {e}")
