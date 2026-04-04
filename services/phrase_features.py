"""
phrase_features.py — Rich per-phrase audio analysis.

Full feature set per phrase:
  max_words          : hard word cap for generation
  max_syllables      : hard syllable cap for generation
  word_length_profile: "short" | "mixed" | "any"
  sustained_vowel_ratio: 0.0–1.0 voiced fraction
  sustain_label      : "high" | "medium" | "low"
  density_label      : "dense" | "mid" | "sparse_sustained" | "sparse_empty"
  literal_weight     : 0.0–1.0 how closely to follow transcript
  vowel_family_hint  : "bright" | "dark" | "neutral" | "mixed"
  pitch_pattern / pitch_symbol
  energy_level, pause_before, pause_after
  confidence, confidence_label
  is_sustained (bool), overflow_flags (list)
"""

import numpy as np
import librosa


# ── Label maps ───────────────────────────────────────────────────────────────

PITCH_SYMBOLS = {
    "rising":   "↗ rising",
    "falling":  "↘ falling",
    "held":     "~ held",
    "flat":     "→ flat",
    "staccato": "↕ staccato",
}

# Bright vs dark vowel families
VOWEL_FAMILY_BRIGHTNESS = {
    "ayy": "bright",
    "ee":  "bright",
    "ii":  "bright",
    "oh":  "dark",
    "ah":  "dark",
    "uh":  "dark",
}

# Default ruleset caps — matches the spec's "strong default ruleset"
_CAPS = {
    "dense":           {"max_words": 7,  "max_syllables": 10},
    "mid":             {"max_words": 5,  "max_syllables": 8},
    "sparse_sustained": {"max_words": 3, "max_syllables": 5},
    "sparse_empty":    {"max_words": 5,  "max_syllables": 7},
}
_LITERAL_WEIGHTS = {"high": 0.90, "med": 0.55, "low": 0.20}


def extract_phrase_features(
    y: np.ndarray,
    sr: int,
    phrase_map: list,
    onset_times: np.ndarray,
    duration: float,
    global_vowel_family: str = None,
) -> list:
    """
    Augment each phrase with the full feature set.
    Mutates phrase dicts in-place and returns the list.
    """
    if not phrase_map:
        return phrase_map

    # Pitch contour — computed once for the whole signal
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

    global_rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 1e-6

    for i, phrase in enumerate(phrase_map):
        t_start = float(phrase.get("start_time", 0))
        t_end   = float(phrase.get("end_time", t_start + 0.5))
        p_dur   = max(t_end - t_start, 0.05)

        # ── Pitch ─────────────────────────────────────────────────────────
        pitch = _classify_pitch(f0, voiced_flag, pitch_times, t_start, t_end)
        phrase["pitch_pattern"] = pitch
        phrase["pitch_symbol"]  = PITCH_SYMBOLS.get(pitch, pitch)

        # ── Sustained vowel ratio ──────────────────────────────────────────
        svr = _sustained_vowel_ratio(f0, voiced_flag, pitch_times, t_start, t_end)
        phrase["sustained_vowel_ratio"] = round(svr, 3)
        phrase["sustain_label"] = (
            "high" if svr >= 0.65 else ("medium" if svr >= 0.35 else "low")
        )
        phrase["is_sustained"] = svr >= 0.65 and p_dur >= 0.6

        # ── Onset / rhythm density ─────────────────────────────────────────
        if len(onset_times) > 0:
            window_onsets = onset_times[(onset_times >= t_start) & (onset_times <= t_end)]
            density_val = round(len(window_onsets) / p_dur, 2)
        else:
            density_val = 0.0
        phrase["rhythm_density"] = density_val
        phrase["density_label"]  = _density_label(density_val, svr)

        # ── Energy ────────────────────────────────────────────────────────
        s = max(0, int(t_start * sr))
        e = min(len(y), int(t_end * sr))
        seg = y[s:e]
        if len(seg) > 0:
            seg_rms = float(np.sqrt(np.mean(seg ** 2)))
            ratio = seg_rms / global_rms if global_rms > 0 else 1.0
            phrase["energy_level"] = "high" if ratio >= 1.3 else ("low" if ratio <= 0.7 else "mid")
        else:
            phrase["energy_level"] = "mid"

        # ── Pause spacing ─────────────────────────────────────────────────
        phrase["pause_before"] = round(
            float(t_start) if i == 0
            else max(0.0, t_start - float(phrase_map[i - 1].get("end_time", t_start))),
            2,
        )
        if i < len(phrase_map) - 1:
            phrase["pause_after"] = round(max(0.0, float(phrase_map[i + 1]["start_time"]) - t_end), 2)
        else:
            phrase["pause_after"] = round(max(0.0, duration - t_end), 2)

        # ── Transcript confidence ─────────────────────────────────────────
        word_count = len(phrase.get("words", []))
        conf = _estimate_confidence(word_count, p_dur, density_val, svr)
        phrase["confidence"]       = round(conf, 3)
        phrase["confidence_label"] = "high" if conf >= 0.70 else ("med" if conf >= 0.40 else "low")

        # ── Literal weight (0.0–1.0) ──────────────────────────────────────
        lw = _LITERAL_WEIGHTS.get(phrase["confidence_label"], 0.55)
        # Repetitive filler (captured by very low word count vs duration) → push down
        if word_count > 0 and word_count / p_dur < 0.5 and density_val > 2:
            lw = min(lw, 0.25)
        phrase["literal_weight"] = round(lw, 2)

        # ── Hard caps ─────────────────────────────────────────────────────
        density_label = phrase["density_label"]
        caps = _CAPS.get(density_label, _CAPS["mid"])

        mw = caps["max_words"]
        ms = caps["max_syllables"]

        # Reduce caps for low confidence
        if phrase["confidence_label"] == "low":
            mw = max(2, mw - 1)
            ms = max(3, ms - 1)

        # Reduce further for high sustain
        if phrase["is_sustained"] or phrase["sustain_label"] == "high":
            mw = max(2, mw - 1)
            ms = max(3, ms - 2)

        # Staccato can tolerate +1 syllable
        if pitch == "staccato":
            ms += 1

        phrase["max_words"]     = mw
        phrase["max_syllables"] = ms

        # ── Word length profile ───────────────────────────────────────────
        phrase["word_length_profile"] = _word_length_profile(density_label, pitch)

        # ── Vowel family hint per phrase ──────────────────────────────────
        phrase["vowel_family_hint"] = _vowel_hint_for_phrase(
            phrase.get("text", ""), global_vowel_family, pitch
        )

    return phrase_map


# ── Density label (sparse split) ────────────────────────────────────────────

def _density_label(onsets_per_sec: float, sustained_vowel_ratio: float = 0.0) -> str:
    if onsets_per_sec >= 4.0:
        return "dense"
    if onsets_per_sec >= 2.0:
        return "mid"
    # sparse — distinguish melodically stretched from genuinely empty
    if sustained_vowel_ratio >= 0.50:
        return "sparse_sustained"
    return "sparse_empty"


# ── Pitch classification ─────────────────────────────────────────────────────

def _classify_pitch(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
) -> str:
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
    cv = float(np.std(f0_win)) / mean_f0
    if cv > 0.28:
        return "staccato"
    x = np.arange(len(f0_win), dtype=float)
    slope, _ = np.polyfit(x, f0_win, 1)
    norm = slope / mean_f0
    if norm > 0.003:
        return "rising"
    if norm < -0.003:
        return "falling"
    if cv < 0.05:
        return "held"
    return "flat"


# ── Sustained vowel ratio ────────────────────────────────────────────────────

def _sustained_vowel_ratio(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
) -> float:
    """Fraction of the phrase window that is pitched (voiced)."""
    if len(f0) == 0 or len(voiced_flag) == 0:
        return 0.0
    mask = (times >= t_start) & (times <= t_end)
    total = int(np.sum(mask))
    if total == 0:
        return 0.0
    voiced = int(np.sum(mask & voiced_flag))
    return voiced / total


# ── Word length profile ──────────────────────────────────────────────────────

def _word_length_profile(density_label: str, pitch: str) -> str:
    """
    short  → prefer words ≤2 syllables (dense/staccato)
    mixed  → short + medium words (mid/held)
    any    → no restriction (sparse, melodic)
    """
    if density_label == "dense" or pitch == "staccato":
        return "short"
    if density_label in ("mid",) or pitch in ("held", "rising"):
        return "mixed"
    return "any"


# ── Vowel family hint per phrase ─────────────────────────────────────────────

def _vowel_hint_for_phrase(
    phrase_text: str,
    global_family: str,
    pitch: str,
) -> str:
    """
    Returns "bright" | "dark" | "neutral" | "mixed".

    Logic:
    - If global family maps to a brightness → use that
    - Pitch rising/staccato biases toward bright
    - Pitch falling/held biases toward dark
    - Otherwise neutral
    """
    brightness = VOWEL_FAMILY_BRIGHTNESS.get(global_family) if global_family else None

    if brightness:
        # Confirm or flag if pitch contradicts
        if pitch in ("rising", "staccato") and brightness == "dark":
            return "mixed"  # pitch says bright, family says dark → ambiguous
        if pitch in ("falling", "held") and brightness == "bright":
            return "mixed"
        return brightness

    # No global family — infer from pitch
    if pitch in ("rising", "staccato"):
        return "bright"
    if pitch in ("falling", "held"):
        return "dark"
    return "neutral"


# ── Transcript confidence ────────────────────────────────────────────────────

def _estimate_confidence(
    word_count: int,
    duration: float,
    onset_density: float,
    sustained_vowel_ratio: float,
) -> float:
    """
    Expanded confidence heuristic:
    - No words + onsets → missed (very low)
    - High onset density + low word rate → missed syllables (low)
    - High SVR + low word count → melodic hold misread as speech (boost confidence)
    - Good word rate → high
    """
    if word_count == 0:
        if onset_density > 1 or sustained_vowel_ratio > 0.4:
            # Likely a held note or dense passage Whisper missed
            return 0.12
        return 0.40  # true silence / near-silence

    if duration <= 0:
        return 0.50

    wps = word_count / duration

    # High SVR means it's likely a held melodic phrase, not a missed word
    # Whisper often gets 1–2 words on a held note → that's actually okay
    if sustained_vowel_ratio >= 0.60:
        return max(0.55, min(0.85, wps * 0.35 + 0.50))

    # Dense but low word rate → missed syllables
    if onset_density > 5 and wps < 1.5:
        return 0.22
    if onset_density > 3 and wps < 1.0:
        return 0.32

    if wps >= 2.5:
        return 0.92
    if wps >= 1.5:
        return 0.78
    if wps >= 0.8:
        return 0.62
    return 0.45


# ── Debug summary ────────────────────────────────────────────────────────────

def phrase_debug_summary(phrase_map: list) -> list:
    """JSON-serializable debug rows for the UI audit panel."""
    rows = []
    for i, p in enumerate(phrase_map):
        # Build overflow flags for UI warnings
        flags = []
        if p.get("is_sustained"):
            flags.append("sustained")
        if p.get("confidence_label") == "low":
            flags.append("low-conf")
        dl = p.get("density_label", "")
        if dl == "sparse_sustained":
            flags.append("sparse-sustained")
        if dl == "sparse_empty":
            flags.append("sparse-empty")

        rows.append({
            "bar":                  i + 1,
            "text":                 p.get("text", ""),
            "syllables":            p.get("syllables", 0),
            "duration":             round(float(p.get("end_time", 0)) - float(p.get("start_time", 0)), 2),
            "pitch":                p.get("pitch_symbol", "—"),
            "density":              p.get("density_label", "—"),
            "energy":               p.get("energy_level", "—"),
            "confidence":           p.get("confidence_label", "—"),
            "confidence_score":     p.get("confidence", 0),
            "pause_after":          p.get("pause_after", 0),
            "max_words":            p.get("max_words", "—"),
            "max_syllables":        p.get("max_syllables", "—"),
            "literal_weight":       p.get("literal_weight", "—"),
            "sustain_label":        p.get("sustain_label", "—"),
            "sustained_vowel_ratio": p.get("sustained_vowel_ratio", 0),
            "vowel_family_hint":    p.get("vowel_family_hint", "—"),
            "word_length_profile":  p.get("word_length_profile", "—"),
            "flags":                flags,
        })
    return rows
