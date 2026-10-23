"""Local-first feedback history for accepted and edited lyric lines."""

import json
import os
import time
import uuid
from pathlib import Path


PROFILE_DIR = Path(os.getenv("LYRIC_GOAT_DATA_DIR", "data")) / "profile"
FEEDBACK_FILE = PROFILE_DIR / "writing_feedback.jsonl"


def record_feedback(data: dict) -> dict:
    action = str(data.get("action", "accepted")).strip().lower()
    if action not in {"accepted", "edited", "rejected"}:
        raise ValueError("action must be accepted, edited, or rejected")

    entry = {
        "id": uuid.uuid4().hex,
        "action": action,
        "song_id": str(data.get("song_id", ""))[:64],
        "section_id": str(data.get("section_id", ""))[:64],
        "source_line": str(data.get("source_line", ""))[:2000],
        "final_line": str(data.get("final_line", ""))[:2000],
        "metadata": data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
        "created_at": round(time.time(), 3),
    }
    if not entry["final_line"].strip():
        raise ValueError("final_line is required")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def recent_feedback(limit: int = 50) -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    lines = FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-max(1, min(limit, 200)):]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))
