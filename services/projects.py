"""
Song Workspace persistence.

Each song is stored as a single JSON file in data/songs/<id>.json.
A song holds ordered sections (verse, hook, bridge, etc.), and each section
keeps its generated lyrics, the analysis it came from, and an optional copy
of the hum audio so it can be replayed while building the arrangement.

No database — plain JSON on disk. Simple, portable, easy to back up, and
works the same whether running locally or deployed.
"""
import os
import json
import time
import uuid
import shutil
import tempfile
from pathlib import Path

DATA_DIR  = Path(os.getenv("LYRIC_GOAT_DATA_DIR", "data"))
SONGS_DIR = DATA_DIR / "songs"
AUDIO_DIR = DATA_DIR / "audio"

SONGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Section types offered in the arrangement, in typical song order.
SECTION_TYPES = ["intro", "verse", "pre-chorus", "hook", "bridge", "outro"]


def _now() -> float:
    return round(time.time(), 3)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _song_path(song_id: str) -> Path:
    return SONGS_DIR / f"{song_id}.json"


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON via a temp file + rename so a crash can't corrupt the song."""
    tmp = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8"
    )
    try:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def _load(song_id: str) -> dict | None:
    path = _song_path(song_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Songs ─────────────────────────────────────────────────────────────────────

def list_songs() -> list[dict]:
    """Lightweight list for the song picker — no section payloads."""
    songs = []
    for path in SONGS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        songs.append({
            "id":            s.get("id"),
            "title":         s.get("title", "Untitled"),
            "bpm":           s.get("bpm", 90),
            "key":           s.get("key", "auto"),
            "genre":         s.get("genre", "hiphop"),
            "section_count": len(s.get("sections", [])),
            "created_at":    s.get("created_at", 0),
            "updated_at":    s.get("updated_at", 0),
        })
    songs.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return songs


def create_song(title: str, bpm: int = 90, key: str = "auto", genre: str = "hiphop") -> dict:
    song = {
        "id":         _new_id(),
        "title":      (title or "Untitled").strip()[:120],
        "bpm":        int(bpm) if bpm else 90,
        "key":        key or "auto",
        "genre":      genre or "hiphop",
        "sections":   [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    _atomic_write(_song_path(song["id"]), song)
    return song


def get_song(song_id: str) -> dict | None:
    return _load(song_id)


def update_song(song_id: str, **fields) -> dict | None:
    song = _load(song_id)
    if song is None:
        return None
    for k in ("title", "bpm", "key", "genre"):
        if k in fields and fields[k] is not None:
            song[k] = fields[k]
    song["updated_at"] = _now()
    _atomic_write(_song_path(song_id), song)
    return song


def delete_song(song_id: str) -> bool:
    song = _load(song_id)
    if song is None:
        return False
    # Clean up any section audio belonging to this song
    for section in song.get("sections", []):
        _delete_section_audio(section)
    path = _song_path(song_id)
    if path.exists():
        path.unlink()
    return True


# ── Sections ────────────────────────────────────────────────────────────────

def _section_defaults() -> dict:
    return {
        "id":         _new_id(),
        "type":       "verse",
        "label":      "Verse",
        "lyrics":     "",
        "rough_text": "",
        "phrase_map": [],
        "versions":   [],
        "settings":   {},
        "audio_file": None,
        "order":      0,
        "updated_at": _now(),
    }


def add_section(song_id: str, section_data: dict) -> dict | None:
    song = _load(song_id)
    if song is None:
        return None
    section = _section_defaults()
    for k in ("type", "label", "lyrics", "rough_text", "phrase_map",
              "versions", "settings", "audio_file"):
        if k in section_data and section_data[k] is not None:
            section[k] = section_data[k]
    section["order"] = len(song["sections"])
    section["updated_at"] = _now()
    song["sections"].append(section)
    song["updated_at"] = _now()
    _atomic_write(_song_path(song_id), song)
    return section


def update_section(song_id: str, section_id: str, section_data: dict) -> dict | None:
    song = _load(song_id)
    if song is None:
        return None
    for section in song["sections"]:
        if section["id"] == section_id:
            for k in ("type", "label", "lyrics", "rough_text", "phrase_map",
                      "versions", "settings", "audio_file", "order"):
                if k in section_data and section_data[k] is not None:
                    section[k] = section_data[k]
            section["updated_at"] = _now()
            song["updated_at"] = _now()
            _atomic_write(_song_path(song_id), song)
            return section
    return None


def delete_section(song_id: str, section_id: str) -> bool:
    song = _load(song_id)
    if song is None:
        return False
    before = len(song["sections"])
    kept = []
    for section in song["sections"]:
        if section["id"] == section_id:
            _delete_section_audio(section)
        else:
            kept.append(section)
    song["sections"] = kept
    # Re-pack order indices
    for i, s in enumerate(song["sections"]):
        s["order"] = i
    song["updated_at"] = _now()
    _atomic_write(_song_path(song_id), song)
    return len(kept) < before


def reorder_sections(song_id: str, ordered_ids: list[str]) -> dict | None:
    song = _load(song_id)
    if song is None:
        return None
    index = {sid: i for i, sid in enumerate(ordered_ids)}
    song["sections"].sort(key=lambda s: index.get(s["id"], 999))
    for i, s in enumerate(song["sections"]):
        s["order"] = i
    song["updated_at"] = _now()
    _atomic_write(_song_path(song_id), song)
    return song


# ── Section audio ──────────────────────────────────────────────────────────

def save_section_audio(song_id: str, section_id: str, src_path: str, ext: str) -> str | None:
    """Copy a hum recording into permanent storage and link it to the section."""
    song = _load(song_id)
    if song is None:
        return None
    ext = (ext or "webm").lstrip(".")
    filename = f"{song_id}_{section_id}.{ext}"
    dest = AUDIO_DIR / filename
    shutil.copyfile(src_path, dest)
    update_section(song_id, section_id, {"audio_file": filename})
    return filename


def audio_path(filename: str) -> Path | None:
    if not filename:
        return None
    p = AUDIO_DIR / filename
    return p if p.exists() else None


def _delete_section_audio(section: dict) -> None:
    fn = section.get("audio_file")
    if fn:
        p = AUDIO_DIR / fn
        if p.exists():
            p.unlink(missing_ok=True)


# ── Full-song assembly ──────────────────────────────────────────────────────

def assemble_song_text(song_id: str) -> str:
    """Plain-text export of the whole arrangement in order."""
    song = _load(song_id)
    if song is None:
        return ""
    blocks = [f"# {song.get('title', 'Untitled')}"]
    meta = f"{song.get('bpm', '?')} BPM · {song.get('key', 'auto')} · {song.get('genre', '')}"
    blocks.append(meta)
    blocks.append("")
    for section in song.get("sections", []):
        label = section.get("label") or section.get("type", "Section").title()
        blocks.append(f"[{label}]")
        blocks.append(section.get("lyrics", "").strip() or "(empty)")
        blocks.append("")
    return "\n".join(blocks).strip()
