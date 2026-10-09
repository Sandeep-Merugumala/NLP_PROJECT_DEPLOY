"""
Unit tests for reward event logging and reading history.
Tests the append_reward_event helper and the /api/history response format.
Run: python -m pytest backend/tests/test_history.py -v
"""

import sys
import json
import tempfile
import os
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------- Standalone helper mirrors app.py ----------
def append_reward_event(log_file, event_type, paper_id, value=None):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "paper_id": paper_id,
        "value": value
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_history(log_file, limit=50):
    events = []
    if not os.path.exists(log_file):
        return events
    with open(log_file, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line.strip())
            events.append(entry)
            if len(events) >= limit:
                break
        except Exception:
            continue
    return events


# ---------- Tests ----------

def test_single_event_is_logged():
    """Logging one event should produce one line in the JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        append_reward_event(path, "upvote", "paper_001", 1)
        with open(path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 1
    finally:
        os.unlink(path)


def test_event_has_required_fields():
    """Each logged event must have timestamp, event_type, paper_id, value."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        append_reward_event(path, "bookmark", "paper_002", None)
        with open(path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert "event_type" in entry
        assert "paper_id" in entry
        assert "value" in entry
        assert entry["event_type"] == "bookmark"
        assert entry["paper_id"] == "paper_002"
    finally:
        os.unlink(path)


def test_multiple_events_logged_in_order():
    """Multiple events should appear in chronological (append) order."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        for i, etype in enumerate(["upvote", "downvote", "bookmark"]):
            append_reward_event(path, etype, f"paper_{i}", i)
        with open(path) as f:
            entries = [json.loads(l) for l in f.readlines()]
        event_types = [e["event_type"] for e in entries]
        assert event_types == ["upvote", "downvote", "bookmark"]
    finally:
        os.unlink(path)


def test_read_history_returns_reversed_order():
    """read_history should return most-recent events first."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        append_reward_event(path, "upvote", "paper_A", 1)
        append_reward_event(path, "downvote", "paper_B", -1)
        append_reward_event(path, "bookmark", "paper_C", None)

        events = read_history(path, limit=50)
        assert events[0]["event_type"] == "bookmark"  # most recent first
        assert events[-1]["event_type"] == "upvote"
    finally:
        os.unlink(path)


def test_history_limit_respected():
    """read_history should not return more than `limit` events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        for i in range(20):
            append_reward_event(path, "upvote", f"paper_{i}", 1)
        events = read_history(path, limit=5)
        assert len(events) == 5
    finally:
        os.unlink(path)


def test_missing_log_file_returns_empty():
    """read_history on a non-existent file should return []."""
    events = read_history("/tmp/__does_not_exist_xyz__.jsonl", limit=50)
    assert events == []


def test_timestamp_is_iso_format():
    """Timestamp should be parseable as ISO 8601."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        append_reward_event(path, "upvote", "paper_X", 1)
        with open(path) as f:
            entry = json.loads(f.readline())
        ts = entry["timestamp"].rstrip("Z")
        parsed = datetime.datetime.fromisoformat(ts)  # should not raise
        assert parsed is not None
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_single_event_is_logged()
    test_event_has_required_fields()
    test_multiple_events_logged_in_order()
    test_read_history_returns_reversed_order()
    test_history_limit_respected()
    test_missing_log_file_returns_empty()
    test_timestamp_is_iso_format()
    print("All history/logging tests passed!")
