"""Optional melody-note providers.

Librosa/pYIN remains the zero-install default. Spotify Basic Pitch can be
enabled for local benchmarking with ``MELODY_NOTE_PROVIDER=basic_pitch`` after
installing ``requirements-melody.txt``. No network API or paid credits are used.
"""

from __future__ import annotations

import os


class MelodyProviderError(RuntimeError):
    pass


def selected_provider() -> str:
    return os.getenv("MELODY_NOTE_PROVIDER", "pyin").strip().lower()


def basic_pitch_notes(audio_path: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise MelodyProviderError(
            "Basic Pitch is not installed. Run: pip install -r requirements-melody.txt"
        ) from exc

    _, _, events = predict(audio_path)
    notes = []
    for event in events:
        start, end, pitch, amplitude = event[:4]
        notes.append({
            "start_time": round(float(start), 3),
            "end_time": round(float(end), 3),
            "duration": round(float(end) - float(start), 3),
            "pitch_midi": int(round(float(pitch))),
            "pitch_name": "",
            "amplitude": round(float(amplitude), 4),
        })
    return notes
