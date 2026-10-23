"""Speech-transcription adapters with explicit model diagnostics.

OpenAI's GPT-4o transcription models only return JSON and do not expose word
timestamps. Whisper-1 supports verbose JSON with word timestamps. Lyric Goat
therefore exposes three deliberate modes instead of silently falling back:

``timed`` (default)
    One Whisper-1 call. Cheapest option that preserves word timing.
``semantic``
    One GPT-4o Transcribe call. Better rough text, while cadence timing comes
    from Lyric Goat's audio-analysis branch.
``hybrid``
    Both calls. Best information, but roughly doubles transcription usage.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

WHISPER_PROMPT = (
    "hip hop rap lyrics mumble singing vocal melody hook verse chorus "
    "yeah ayy ay aye ooh oh woah nah gonna wanna tryna gotta "
    "mm hmm ah la da ba bo doo woo hey yo na na "
    "singing humming melody vocal run riff ad-lib"
)

SEMANTIC_MODEL = os.getenv("OPENAI_SEMANTIC_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")
TIMING_MODEL = os.getenv("OPENAI_TIMING_TRANSCRIPTION_MODEL", "whisper-1")
DEFAULT_MODE = os.getenv("LYRIC_GOAT_TRANSCRIPTION_MODE", "timed").lower()
VALID_MODES = {"timed", "semantic", "hybrid"}


class TranscriptionError(RuntimeError):
    """Raised when a requested transcription provider fails."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TranscriptionError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_path: str, mode: str | None = None) -> dict:
    """Transcribe audio in an explicit cost/quality mode."""
    selected_mode = (mode or DEFAULT_MODE).lower()
    if selected_mode not in VALID_MODES:
        raise TranscriptionError(
            f"Unknown transcription mode '{selected_mode}'. Expected one of: "
            + ", ".join(sorted(VALID_MODES))
        )

    diagnostics: dict[str, Any] = {
        "mode": selected_mode,
        "semantic_model": None,
        "timing_model": None,
        "errors": [],
    }
    semantic: dict | None = None
    timed: dict | None = None

    if selected_mode in {"semantic", "hybrid"}:
        try:
            semantic = _transcribe_semantic(audio_path)
            diagnostics["semantic_model"] = SEMANTIC_MODEL
        except Exception as exc:
            diagnostics["errors"].append({
                "stage": "semantic",
                "model": SEMANTIC_MODEL,
                "message": _safe_error(exc),
            })
            logger.error("Semantic transcription failed with %s: %s", SEMANTIC_MODEL, _safe_error(exc))

    if selected_mode in {"timed", "hybrid"}:
        try:
            timed = _transcribe_timed(audio_path)
            diagnostics["timing_model"] = TIMING_MODEL
        except Exception as exc:
            diagnostics["errors"].append({
                "stage": "timing",
                "model": TIMING_MODEL,
                "message": _safe_error(exc),
            })
            logger.error("Timed transcription failed with %s: %s", TIMING_MODEL, _safe_error(exc))

    if semantic is None and timed is None:
        raise TranscriptionError("All requested transcription stages failed", diagnostics)

    diagnostics["status"] = "partial" if diagnostics["errors"] else "ok"
    return {
        "text": (semantic or timed or {}).get("text", ""),
        "words": (timed or {}).get("words", []),
        "duration": (timed or semantic or {}).get("duration"),
        "diagnostics": diagnostics,
    }


def _transcribe_semantic(audio_path: str) -> dict:
    """Use GPT-4o Transcribe with the JSON-only response it supports."""
    with open(audio_path, "rb") as audio_file:
        response = _client().audio.transcriptions.create(
            model=SEMANTIC_MODEL,
            file=audio_file,
            response_format="json",
            language="en",
            prompt=WHISPER_PROMPT,
            include=["logprobs"],
        )

    return {
        "text": getattr(response, "text", "").strip(),
        "words": [],
        "duration": _usage_duration(getattr(response, "usage", None)),
        "average_logprob": _average_logprob(getattr(response, "logprobs", None) or []),
    }


def _transcribe_timed(audio_path: str) -> dict:
    """Use Whisper's verbose response for word-level timestamps."""
    with open(audio_path, "rb") as audio_file:
        response = _client().audio.transcriptions.create(
            model=TIMING_MODEL,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
            prompt=WHISPER_PROMPT,
            temperature=0.1,
        )

    words = []
    for word in getattr(response, "words", None) or []:
        words.append({
            "word": getattr(word, "word", ""),
            "start": float(getattr(word, "start", 0.0)),
            "end": float(getattr(word, "end", 0.0)),
        })

    return {
        "text": getattr(response, "text", "").strip(),
        "words": words,
        "duration": getattr(response, "duration", None),
    }


def _average_logprob(logprobs: list) -> float | None:
    values = []
    for item in logprobs:
        value = getattr(item, "logprob", None)
        if value is None and isinstance(item, dict):
            value = item.get("logprob")
        if value is not None:
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else None


def _usage_duration(usage: Any) -> float | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("seconds")
    return getattr(usage, "seconds", None)


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]
