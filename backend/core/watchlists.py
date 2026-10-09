"""
Watchlists and Alerts — Part 2 Feature #6
Follow keywords, authors, and venues. Get new-paper feed with notifications
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class WatchType(Enum):
    """Types of things to watch"""
    KEYWORD = "keyword"
    AUTHOR = "author"
    VENUE = "venue"
    TOPIC = "topic"
    RESEARCH_AREA = "research_area"

@dataclass
class Watchlist:
    """User's watchlist"""
    watchlist_id: str
    name: str
    description: str
    items: List[Dict]  # {type, value, added_at, enabled}
    created_at: str
    updated_at: str

@dataclass
class Alert:
    """Notification alert"""
    alert_id: str
    watchlist_id: str
    paper_id: str
    paper_title: str
    watch_item: str  # Which item matched
    watch_type: str
    match_strength: float  # 0.0-1.0
    created_at: str
    is_read: bool

class WatchlistManager:
    """Manage watchlists and generate alerts"""
    
    def __init__(self, embeddings_model, llm_router, storage_path: str = "data/watchlists.json"):
        self.embeddings = embeddings_model
        self.llm = llm_router
        self.storage_path = storage_path
        self.watchlists = self._load_watchlists()
        self.alerts = self._load_alerts()
        self.keyword_embeddings_cache = {}
    
    def create_watchlist(
        self,
        name: str,
        description: str = ""
    ) -> Watchlist:
        """Create a new watchlist"""
        
        watchlist_id = str(uuid.uuid4())[:8]
        watchlist = Watchlist(
            watchlist_id=watchlist_id,
            name=name,
            description=description,
            items=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.watchlists[watchlist_id] = watchlist
        self._save_watchlists()
        
        return watchlist
    
    def add_watch_item(
        self,
        watchlist_id: str,
        item_type: str,
        value: str
    ) -> bool:
        """Add an item to watch (keyword, author, venue, etc.)"""
        
        if watchlist_id not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_id]
        
        # Check if already watching
        if any(item["value"] == value and item["type"] == item_type for item in watchlist.items):
            return False
        
        watchlist.items.append({
            "type": item_type,
            "value": value,
            "added_at": datetime.now().isoformat(),
            "enabled": True,
            "hit_count": 0
        })
        
        watchlist.updated_at = datetime.now().isoformat()
        self._save_watchlists()
        
        # Cache embedding for keywords
        if item_type == WatchType.KEYWORD.value:
            self.keyword_embeddings_cache[value] = self.embeddings.encode(value)
        
        return True
    
    def remove_watch_item(
        self,
        watchlist_id: str,
        item_type: str,
        value: str
    ) -> bool:
        """Remove an item from watchlist"""
        
        if watchlist_id not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_id]
        original_len = len(watchlist.items)
        
        watchlist.items = [
            item for item in watchlist.items
            if not (item["type"] == item_type and item["value"] == value)
        ]
        
        if len(watchlist.items) < original_len:
            watchlist.updated_at = datetime.now().isoformat()
            self._save_watchlists()
            return True
        
        return False
    
    def toggle_watch_item(
        self,
        watchlist_id: str,
        item_type: str,
        value: str,
        enabled: bool
    ) -> bool:
        """Enable/disable a watch item"""
        
        if watchlist_id not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_id]
        
        for item in watchlist.items:
            if item["type"] == item_type and item["value"] == value:
                item["enabled"] = enabled
                watchlist.updated_at = datetime.now().isoformat()
                self._save_watchlists()
                return True
        
        return False
    
    def process_new_papers(
        self,
        new_papers: List[Dict],
        watchlist_ids: Optional[List[str]] = None
    ) -> List[Alert]:
        """Process new papers and generate alerts for matching watchlists"""
        
        if not watchlist_ids:
            watchlist_ids = list(self.watchlists.keys())
        
        new_alerts = []
        
        for watchlist_id in watchlist_ids:
            if watchlist_id not in self.watchlists:
                continue
            
            watchlist = self.watchlists[watchlist_id]
            enabled_items = [item for item in watchlist.items if item["enabled"]]
            
            # Build a set of already-alerted (paper_id, watch_item) pairs for this watchlist
            existing_pairs = {
                (a.paper_id, a.watch_item)
                for a in self.alerts.values()
                if a.watchlist_id == watchlist_id
            }

            for paper in new_papers:
                for item in enabled_items:
                    # Skip if we already have an alert for this paper+item (avoids re-alerting after mark-as-read)
                    if (paper["id"], item["value"]) in existing_pairs:
                        continue

                    match_strength = self._calculate_match_strength(paper, item)
                    
                    if match_strength > 0.4:  # Threshold
                        alert = Alert(
                            alert_id=str(uuid.uuid4())[:8],
                            watchlist_id=watchlist_id,
                            paper_id=paper["id"],
                            paper_title=paper.get("title", "Untitled"),
                            watch_item=item["value"],
                            watch_type=item["type"],
                            match_strength=match_strength,
                            created_at=datetime.now().isoformat(),
                            is_read=False
                        )
                        
                        new_alerts.append(alert)
                        self.alerts[alert.alert_id] = alert
                        
                        # Update hit count
                        item["hit_count"] = item.get("hit_count", 0) + 1
        
        self._save_alerts()
        self._save_watchlists()
        
        return new_alerts
    
    def _calculate_match_strength(self, paper: Dict, watch_item: Dict) -> float:
        """Calculate how strongly a paper matches a watch item"""
        
        item_type = watch_item["type"]
        item_value = watch_item["value"].lower()
        
        if item_type == WatchType.KEYWORD.value:
            # Keyword matching via semantic similarity
            return self._keyword_match_strength(
                paper.get("title", "") + " " + paper.get("abstract", ""),
                item_value
            )
        
        elif item_type == WatchType.AUTHOR.value:
            # Check if author is in the paper
            authors = [a.get("name", "").lower() for a in paper.get("authors", [])]
            if any(item_value in author for author in authors):
                return 1.0
            return 0.0
        
        elif item_type == WatchType.VENUE.value:
            # Check if venue matches
            venue = paper.get("venue", "").lower()
            if item_value in venue:
                return 1.0
            return 0.0
        
        elif item_type == WatchType.TOPIC.value:
            # Topic matching via embedding
            return self._topic_match_strength(
                paper.get("title", "") + " " + paper.get("abstract", ""),
                item_value
            )
        
        elif item_type == WatchType.RESEARCH_AREA.value:
            # Research area in paper tags/categories
            areas = paper.get("research_areas", [])
            if any(item_value.lower() in area.lower() for area in areas):
                return 1.0
            return 0.0
        
        return 0.0
    
    def _keyword_match_strength(self, paper_text: str, keyword: str) -> float:
        """Semantic similarity for keyword matching"""
        
        if keyword not in self.keyword_embeddings_cache:
            self.keyword_embeddings_cache[keyword] = self.embeddings.encode(keyword)
        
        keyword_embedding = self.keyword_embeddings_cache[keyword]
        paper_embedding = self.embeddings.encode(paper_text[:1000])  # First 1000 chars
        
        similarity = float(np.dot(keyword_embedding, paper_embedding))
        return max(0.0, min(1.0, similarity))  # Normalize to 0-1
    
    def _topic_match_strength(self, paper_text: str, topic: str) -> float:
        """Semantic similarity for topic matching"""
        
        # Similar to keyword matching
        return self._keyword_match_strength(paper_text, topic)
    
    def get_unread_alerts(
        self,
        watchlist_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get unread alerts"""
        
        alerts = [
            asdict(alert) for alert in self.alerts.values()
            if not alert.is_read and (watchlist_id is None or alert.watchlist_id == watchlist_id)
        ]
        
        # Sort by creation time, newest first
        alerts.sort(key=lambda a: a["created_at"], reverse=True)
        
        return alerts[:limit]
    
    def mark_alert_read(self, alert_id: str) -> bool:
        """Mark an alert as read"""
        
        if alert_id not in self.alerts:
            return False
        
        self.alerts[alert_id].is_read = True
        self._save_alerts()
        return True
    
    def mark_all_read(self, watchlist_id: Optional[str] = None) -> int:
        """Mark all alerts as read"""
        
        count = 0
        for alert in self.alerts.values():
            if not alert.is_read and (watchlist_id is None or alert.watchlist_id == watchlist_id):
                alert.is_read = True
                count += 1
        
        self._save_alerts()
        return count
    
    def get_alert_summary(self, days: int = 7) -> Dict:
        """Get summary of alerts in the past N days"""
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.alerts.values()
            if datetime.fromisoformat(alert.created_at) > cutoff
        ]
        
        # Group by watchlist
        by_watchlist = {}
        for alert in recent_alerts:
            if alert.watchlist_id not in by_watchlist:
                by_watchlist[alert.watchlist_id] = {"read": 0, "unread": 0}
            
            if alert.is_read:
                by_watchlist[alert.watchlist_id]["read"] += 1
            else:
                by_watchlist[alert.watchlist_id]["unread"] += 1
        
        # Get top matching items
        item_hits = {}
        for alert in recent_alerts:
            key = (alert.watch_type, alert.watch_item)
            if key not in item_hits:
                item_hits[key] = 0
            item_hits[key] += 1
        
        top_items = sorted(item_hits.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "period_days": days,
            "total_alerts": len(recent_alerts),
            "by_watchlist": by_watchlist,
            "top_matching_items": [
                {"type": t, "value": v, "hits": h}
                for (t, v), h in top_items
            ]
        }
    
    def get_watchlist_stats(self, watchlist_id: str) -> Dict:
        """Get statistics for a watchlist"""
        
        if watchlist_id not in self.watchlists:
            return None
        
        watchlist = self.watchlists[watchlist_id]
        
        watchlist_alerts = [
            alert for alert in self.alerts.values()
            if alert.watchlist_id == watchlist_id
        ]
        
        return {
            "watchlist_id": watchlist_id,
            "watchlist_name": watchlist.name,
            "num_items": len(watchlist.items),
            "total_alerts": len(watchlist_alerts),
            "unread_alerts": sum(1 for a in watchlist_alerts if not a.is_read),
            "items": watchlist.items,
            "created_at": watchlist.created_at,
            "last_updated": watchlist.updated_at
        }
    
    def _load_watchlists(self) -> Dict[str, Watchlist]:
        """Load watchlists from storage"""
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {
                    wid: Watchlist(**wdata)
                    for wid, wdata in data.get("watchlists", {}).items()
                }
        except:
            return {}
    
    def _load_alerts(self) -> Dict[str, Alert]:
        """Load alerts from storage"""
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {
                    aid: Alert(**adata)
                    for aid, adata in data.get("alerts", {}).items()
                }
        except:
            return {}
    
    def _save_watchlists(self):
        """Persist watchlists"""
        
        data = json.load(open(self.storage_path)) if os.path.exists(self.storage_path) else {"alerts": {}}
        data["watchlists"] = {
            wid: asdict(watchlist)
            for wid, watchlist in self.watchlists.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_alerts(self):
        """Persist alerts"""
        
        data = json.load(open(self.storage_path)) if os.path.exists(self.storage_path) else {"watchlists": {}}
        data["alerts"] = {
            aid: asdict(alert)
            for aid, alert in self.alerts.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# For brevity, import os at top
import os
