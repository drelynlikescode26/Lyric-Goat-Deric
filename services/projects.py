"""
Song Workspace persistence — Supabase backend.

Songs and their sections live in Postgres (tables: songs, sections); hum audio
lives in the `section-audio` storage bucket. This replaces the old JSON-on-disk
store so data persists across deploys and stays in sync between phone and
desktop.

Configuration (env):
  SUPABASE_URL   e.g. https://xxxx.supabase.co
  SUPABASE_KEY   publishable / anon key

The public function API is unchanged from the JSON version, so app.py needs
no changes beyond how section audio is served (now a redirect to the bucket).
"""
import os
from collections import Counter
from datetime import datetime, timezone

from supabase import create_client, Client

BUCKET = "section-audio"

_client: Client | None = None

_MIME = {
    "webm": "audio/webm", "mp4": "audio/mp4", "m4a": "audio/mp4",
    "ogg": "audio/ogg", "mp3": "audio/mpeg", "wav": "audio/wav",
}


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "Supabase not configured — set SUPABASE_URL and SUPABASE_KEY "
                "in your environment (.env)."
            )
        _client = create_client(url, key)
    return _client


# ── Serializers (DB row → API shape) ─────────────────────────────────────────

def _section_out(row: dict) -> dict:
    return {
        "id":         row["id"],
        "type":       row["type"],
        "label":      row["label"],
        "lyrics":     row["lyrics"],
        "rough_text": row["rough_text"],
        "phrase_map": row["phrase_map"],
        "versions":   row["versions"],
        "settings":   row["settings"],
        "audio_file": row["audio_file"],
        "order":      row["order_index"],
        "updated_at": row["updated_at"],
    }


def _song_out(row: dict, sections: list[dict]) -> dict:
    return {
        "id":         row["id"],
        "title":      row["title"],
        "bpm":        row["bpm"],
        "key":        row["key"],
        "genre":      row["genre"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "sections":   sections,
    }


# ── Songs ─────────────────────────────────────────────────────────────────────

def list_songs() -> list[dict]:
    c = get_client()
    songs = c.table("songs").select("*").order("updated_at", desc=True).execute().data or []
    sec_rows = c.table("sections").select("song_id").execute().data or []
    counts = Counter(r["song_id"] for r in sec_rows)
    return [{
        "id":            s["id"],
        "title":         s["title"],
        "bpm":           s["bpm"],
        "key":           s["key"],
        "genre":         s["genre"],
        "section_count": counts.get(s["id"], 0),
        "created_at":    s["created_at"],
        "updated_at":    s["updated_at"],
    } for s in songs]


def create_song(title: str, bpm: int = 90, key: str = "auto", genre: str = "hiphop") -> dict:
    c = get_client()
    row = c.table("songs").insert({
        "title": (title or "Untitled").strip()[:120],
        "bpm":   int(bpm) if bpm else 90,
        "key":   key or "auto",
        "genre": genre or "hiphop",
    }).execute().data[0]
    return _song_out(row, [])


def get_song(song_id: str) -> dict | None:
    c = get_client()
    songs = c.table("songs").select("*").eq("id", song_id).limit(1).execute().data
    if not songs:
        return None
    sections = c.table("sections").select("*").eq("song_id", song_id)\
        .order("order_index").execute().data or []
    return _song_out(songs[0], [_section_out(s) for s in sections])


def update_song(song_id: str, **fields) -> dict | None:
    c = get_client()
    payload = {k: fields[k] for k in ("title", "bpm", "key", "genre")
               if k in fields and fields[k] is not None}
    if payload:
        c.table("songs").update(payload).eq("id", song_id).execute()
    return get_song(song_id)


def delete_song(song_id: str) -> bool:
    c = get_client()
    # Remove any stored audio for this song's sections first
    sections = c.table("sections").select("audio_file").eq("song_id", song_id).execute().data or []
    paths = [s["audio_file"] for s in sections if s.get("audio_file")]
    if paths:
        try:
            c.storage.from_(BUCKET).remove(paths)
        except Exception:
            pass
    c.table("songs").delete().eq("id", song_id).execute()  # cascades to sections
    return True


# ── Sections ────────────────────────────────────────────────────────────────

_SECTION_FIELDS = ("type", "label", "lyrics", "rough_text",
                   "phrase_map", "versions", "settings", "audio_file")


def add_section(song_id: str, section_data: dict) -> dict | None:
    c = get_client()
    if not c.table("songs").select("id").eq("id", song_id).limit(1).execute().data:
        return None
    existing = c.table("sections").select("id").eq("song_id", song_id).execute().data or []
    payload = {"song_id": song_id, "order_index": len(existing)}
    for k in _SECTION_FIELDS:
        if k in section_data and section_data[k] is not None:
            payload[k] = section_data[k]
    row = c.table("sections").insert(payload).execute().data[0]
    return _section_out(row)


def update_section(song_id: str, section_id: str, section_data: dict) -> dict | None:
    c = get_client()
    payload = {}
    for k in _SECTION_FIELDS:
        if k in section_data and section_data[k] is not None:
            payload[k] = section_data[k]
    if "order" in section_data and section_data["order"] is not None:
        payload["order_index"] = section_data["order"]
    if payload:
        c.table("sections").update(payload)\
            .eq("id", section_id).eq("song_id", song_id).execute()
        # keep parent song's updated_at fresh so it sorts to the top
        c.table("songs").update({"updated_at": datetime.now(timezone.utc).isoformat()})\
            .eq("id", song_id).execute()
    rows = c.table("sections").select("*").eq("id", section_id).limit(1).execute().data
    return _section_out(rows[0]) if rows else None


def delete_section(song_id: str, section_id: str) -> bool:
    c = get_client()
    rows = c.table("sections").select("audio_file").eq("id", section_id).execute().data or []
    for r in rows:
        if r.get("audio_file"):
            try:
                c.storage.from_(BUCKET).remove([r["audio_file"]])
            except Exception:
                pass
    c.table("sections").delete().eq("id", section_id).eq("song_id", song_id).execute()
    # Re-pack order indices
    remaining = c.table("sections").select("id").eq("song_id", song_id)\
        .order("order_index").execute().data or []
    for i, s in enumerate(remaining):
        c.table("sections").update({"order_index": i}).eq("id", s["id"]).execute()
    return True


def reorder_sections(song_id: str, ordered_ids: list[str]) -> dict | None:
    c = get_client()
    for i, sid in enumerate(ordered_ids):
        c.table("sections").update({"order_index": i})\
            .eq("id", sid).eq("song_id", song_id).execute()
    return get_song(song_id)


# ── Section audio (Supabase Storage) ─────────────────────────────────────────

def save_section_audio(song_id: str, section_id: str, src_path: str, ext: str) -> str | None:
    c = get_client()
    if not c.table("sections").select("id")\
            .eq("id", section_id).eq("song_id", song_id).limit(1).execute().data:
        return None
    ext = (ext or "webm").lstrip(".").lower()
    path = f"{song_id}/{section_id}.{ext}"
    with open(src_path, "rb") as f:
        data = f.read()
    # Overwrite any previous take
    try:
        c.storage.from_(BUCKET).remove([path])
    except Exception:
        pass
    c.storage.from_(BUCKET).upload(
        path, data, {"content-type": _MIME.get(ext, "audio/webm")}
    )
    c.table("sections").update({"audio_file": path})\
        .eq("id", section_id).eq("song_id", song_id).execute()
    return path


def get_audio_public_url(path: str) -> str | None:
    if not path:
        return None
    try:
        return get_client().storage.from_(BUCKET).get_public_url(path)
    except Exception:
        return None


# ── Full-song assembly ──────────────────────────────────────────────────────

def assemble_song_text(song_id: str) -> str:
    song = get_song(song_id)
    if song is None:
        return ""
    blocks = [f"# {song.get('title', 'Untitled')}"]
    blocks.append(f"{song.get('bpm', '?')} BPM · {song.get('key', 'auto')} · {song.get('genre', '')}")
    blocks.append("")
    for section in song.get("sections", []):
        label = section.get("label") or section.get("type", "Section").title()
        blocks.append(f"[{label}]")
        blocks.append(section.get("lyrics", "").strip() or "(empty)")
        blocks.append("")
    return "\n".join(blocks).strip()
