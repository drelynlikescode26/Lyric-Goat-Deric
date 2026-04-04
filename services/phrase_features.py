"""
phrase_features.py — Rich per-phrase audio analysis.

Extracts pitch contour, rhythm density, energy envelope, pause spacing,
and transcript confidence for each phrase in the phrase_map.

These features are used to:
  1. Tell the lyric generator whether a phrase is fast/staccato or held/melodic
  2. Communicate pitch direction (rising/falling/held) so lyrics can mirror the melody
  3. Flag low-confidence phrases so the generator ignores unreliable word guesses
  4. Track pause spacing so generated lines have natural breathing room

All functions accept and return the phrase_map list in-place (mutates + returns).
"""

import numpy as np
import librosa
from typing import Optional


# ── Pitch labels used in generation prompt ──────────────────────────────────

PITCH_SYMBOLS = {
    "rising":   "↗ rising",
    "falling":  "↘ falling",
    "held":     "~ held",
    "flat":     "→ flat",
    "staccato": "↕ staccato",
}

DENSITY_LABELS = {
    "dense":  "dense",    # > 4 onsets/sec
    "mid":    "mid",      # 2–4 onsets/sec
    "sparse": "sparse",   # < 2 onsets/sec
}


def extract_phrase_features(
    y: np.ndarray,
    sr: int,
    phrase_map: list,
    onset_times: np.ndarray,
    duration: float,
) -> list:
    """
    Augment each phrase in phrase_map with rich audio features.
    Mutates the phrase dicts in-place and returns the list.

    Added keys per phrase:
      pitch_pattern   : str  — "rising" | "falling" | "held" | "flat" | "staccato"
      pitch_symbol    : str  — human-readable label e.g. "↗ rising"
      rhythm_density  : float — onsets per second in this phrase window
      density_label   : str  — "dense" | "mid" | "sparse"
      energy_level    : str  — "high" | "mid" | "low"
      pause_before    : float — seconds of silence before phrase
      pause_after     : float — seconds after phrase before next starts
      confidence      : float — 0.0–1.0 how reliably Whisper captured this phrase
      confidence_label: str  — "high" | "med" | "low"
    """
    if not phrase_map:
        return phrase_map

    # Compute pitch contour for the whole signal once
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        pitch_times = librosa.times_like(f0, sr=sr)
    except Exception:
        f0 = np.array([])
        voiced_flag = np.array([], dtype=bool)
        pitch_times = np.array([])

    # Global RMS for relative energy comparison
    global_rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 1e-6

    for i, phrase in enumerate(phrase_map):
        t_start = float(phrase.get("start_time", 0))
        t_end   = float(phrase.get("end_time", t_start + 0.5))
        p_dur   = max(t_end - t_start, 0.05)

        # ── Pitch pattern ─────────────────────────────────────────────────
        pitch = _classify_pitch(f0, voiced_flag, pitch_times, t_start, t_end)
        phrase["pitch_pattern"] = pitch
        phrase["pitch_symbol"]  = PITCH_SYMBOLS.get(pitch, pitch)

        # ── Rhythm density ────────────────────────────────────────────────
        if len(onset_times) > 0:
            window_onsets = onset_times[(onset_times >= t_start) & (onset_times <= t_end)]
            density = round(len(window_onsets) / p_dur, 2)
        else:
            density = 0.0
        phrase["rhythm_density"] = density
        phrase["density_label"]  = _density_label(density)

        # ── Energy level ─────────────────────────────────────────────────
        s = max(0, int(t_start * sr))
        e = min(len(y), int(t_end * sr))
        phrase_y = y[s:e]
        if len(phrase_y) > 0:
            phrase_rms = float(np.sqrt(np.mean(phrase_y ** 2)))
            phrase["energy_level"] = _energy_label(phrase_rms, global_rms)
        else:
            phrase["energy_level"] = "mid"

        # ── Pause spacing ────────────────────────────────────────────────
        if i == 0:
            phrase["pause_before"] = round(t_start, 2)
        else:
            prev_end = float(phrase_map[i - 1].get("end_time", t_start))
            phrase["pause_before"] = round(max(0.0, t_start - prev_end), 2)

        if i < len(phrase_map) - 1:
            next_start = float(phrase_map[i + 1]["start_time"])
            phrase["pause_after"] = round(max(0.0, next_start - t_end), 2)
        else:
            phrase["pause_after"] = round(max(0.0, duration - t_end), 2)

        # ── Transcript confidence ────────────────────────────────────────
        word_count = len(phrase.get("words", []))
        conf = _estimate_confidence(word_count, p_dur, density)
        phrase["confidence"]       = round(conf, 2)
        phrase["confidence_label"] = "high" if conf >= 0.7 else ("med" if conf >= 0.4 else "low")

        # ── Sustained vowel detection ────────────────────────────────────
        phrase["is_sustained"] = _detect_sustained_vowel(
            f0, voiced_flag, pitch_times, t_start, t_end, p_dur
        )

    return phrase_map


# ── Internal helpers ─────────────────────────────────────────────────────────

def _classify_pitch(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
) -> str:
    """Classify pitch motion in a time window using linear slope + variance."""
    if len(f0) == 0 or len(voiced_flag) == 0:
        return "flat"

    mask = (times >= t_start) & (times <= t_end) & voiced_flag
    f0_win = f0[mask]
    f0_win = f0_win[~np.isnan(f0_win)]

    if len(f0_win) < 4:
        return "flat"

    mean_f0 = float(np.mean(f0_win))
    if mean_f0 <= 0:
        return "flat"

    std_f0 = float(np.std(f0_win))
    cv = std_f0 / mean_f0  # coefficient of variation

    # Very jumpy pitch = rhythmic/staccato
    if cv > 0.28:
        return "staccato"

    # Linear slope normalized by mean
    x = np.arange(len(f0_win), dtype=float)
    slope, _ = np.polyfit(x, f0_win, 1)
    norm_slope = slope / mean_f0

    if norm_slope > 0.003:
        return "rising"
    if norm_slope < -0.003:
        return "falling"
    if cv < 0.05:
        return "held"
    return "flat"


def _density_label(onsets_per_sec: float) -> str:
    if onsets_per_sec >= 4.0:
        return "dense"
    if onsets_per_sec >= 2.0:
        return "mid"
    return "sparse"


def _energy_label(phrase_rms: float, global_rms: float) -> str:
    ratio = phrase_rms / global_rms if global_rms > 0 else 1.0
    if ratio >= 1.3:
        return "high"
    if ratio <= 0.7:
        return "low"
    return "mid"


def _estimate_confidence(word_count: int, duration: float, onset_density: float) -> float:
    """
    Estimate how well Whisper captured this phrase.

    Heuristic logic:
    - No words + any onsets → Whisper missed it entirely → very low
    - High onset density but few words → Whisper missed syllables → low
    - Good word rate → high confidence
    """
    if word_count == 0:
        return 0.1 if onset_density > 1 else 0.4  # silence vs missed
    if duration <= 0:
        return 0.5

    words_per_sec = word_count / duration

    # High syllable density but low word count → Whisper missed words
    if onset_density > 5 and words_per_sec < 1.5:
        return 0.25
    if onset_density > 3 and words_per_sec < 1.0:
        return 0.35

    if words_per_sec >= 2.5:
        return 0.92
    if words_per_sec >= 1.5:
        return 0.78
    if words_per_sec >= 0.8:
        return 0.62
    return 0.45


def _detect_sustained_vowel(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
    duration: float,
) -> bool:
    """
    Returns True when the phrase is a held/sustained vowel sound
    (e.g. "ooooh", "ayyyy", long melodic hold).

    Criteria (all must pass):
      - Duration >= 0.6s (long enough to be intentional)
      - Voiced ratio >= 0.65 (mostly pitched, not noisy)
      - Pitch CV < 0.10 (stable — not jumping around)
    """
    if duration < 0.6 or len(f0) == 0 or len(voiced_flag) == 0:
        return False

    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        return False

    voiced_ratio = float(np.mean(voiced_flag[mask]))
    if voiced_ratio < 0.65:
        return False

    f0_win = f0[mask & voiced_flag]
    f0_win = f0_win[~np.isnan(f0_win)]
    if len(f0_win) < 4:
        return False

    mean_f0 = float(np.mean(f0_win))
    if mean_f0 <= 0:
        return False

    cv = float(np.std(f0_win)) / mean_f0
    return cv < 0.10


def phrase_debug_summary(phrase_map: list) -> list:
    """
    Return a list of human-readable debug dicts for the UI debug panel.
    Each dict is safe to JSON-serialize.
    """
    rows = []
    for i, p in enumerate(phrase_map):
        flags = []
        if p.get("is_sustained"):
            flags.append("sustained")
        rows.append({
            "bar":        i + 1,
            "text":       p.get("text", ""),
            "syllables":  p.get("syllables", 0),
            "duration":   round(float(p.get("end_time", 0)) - float(p.get("start_time", 0)), 2),
            "pitch":      p.get("pitch_symbol", "—"),
            "density":    p.get("density_label", "—"),
            "energy":     p.get("energy_level", "—"),
            "confidence": p.get("confidence_label", "—"),
            "pause_after": p.get("pause_after", 0),
            "flags":      flags,
        })
    return rows
