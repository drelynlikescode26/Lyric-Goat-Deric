"""Automatic vocal phrasing and syllable nuclei, independent of pitch changes.

An amplitude nucleus is an estimate, not a decoded word. Pitch events are kept
separately so a run can carry one syllable and repeated notes can carry many.
No timing approval step or model/network call is required here.
"""
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, binary_closing
from scipy.signal import find_peaks


def vocal_phrases(y, sr, tempo=90.0):
    samples = np.asarray(y, dtype=float)
    if not len(samples) or not np.isfinite(samples).all():
        return []
    hop = max(1, round(sr * 0.01))
    rms = librosa.feature.rms(y=samples, frame_length=max(64, round(sr * .025)),
                              hop_length=hop)[0]
    envelope = gaussian_filter1d(rms, 1.3)
    peak = float(envelope.max())
    if peak < 1e-5:
        return []
    # Cap the noise-floor estimate for recordings with no leading silence.
    threshold = max(peak * .035, min(float(np.percentile(envelope, 10)) * 3, peak * .14))
    active = binary_closing(envelope > threshold, structure=np.ones(6))
    edges = np.diff(np.r_[False, active, False].astype(int))
    runs = [(a, b) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
            if (b - a) * hop / sr >= .08]
    if not runs:
        return []
    step = hop / sr
    gap = max(.28, min(.45, 60 / max(float(tempo), 40) * .55))
    groups = [list(runs[0])]
    for start, end in runs[1:]:
        if (start - groups[-1][1]) * step <= gap:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    result = []
    for start, end in groups:
        window = envelope[start:end]
        # Padding catches a nucleus at a hard clip boundary. Shallow amplitude
        # vibrato does not meet the prominence threshold.
        peaks, _ = find_peaks(np.r_[0, window, 0], distance=max(1, round(.11 / step)),
                              prominence=float(window.max()) * .20,
                              height=float(window.max()) * .25)
        nuclei = np.clip(peaks - 1 + start, start, end - 1)
        if not len(nuclei):
            nuclei = np.array([start + int(np.argmax(window))])
        nuclei = np.unique(nuclei)
        # Split long continuous passages at syllable boundaries, not at notes.
        for offset in range(0, len(nuclei), 16):
            part = nuclei[offset:offset + 16]
            a = start if offset == 0 else int((nuclei[offset - 1] + part[0]) / 2)
            last = offset + len(part)
            b = end if last == len(nuclei) else int((part[-1] + nuclei[last]) / 2)
            bounds = [a] + [int((x + z) / 2) for x, z in zip(part, part[1:])] + [b]
            slots = []
            for i, nucleus in enumerate(part):
                energy = float(envelope[nucleus] / peak)
                slots.append({"start_time": round(float(bounds[i] * step), 3),
                              "end_time": round(float(min(bounds[i + 1] * step, len(y) / sr)), 3),
                              "nucleus_time": round(float(nucleus * step), 3),
                              "energy": round(energy, 3),
                              "accent": energy >= .80,
                              "vowel": "unknown"})
            count = len(slots)
            result.append({"text": "", "words": [], "syllables": count,
                           "target_syllables": count, "syllable_slots": slots,
                           "start_time": slots[0]["start_time"],
                           "end_time": slots[-1]["end_time"],
                           "timing_source": "acoustic_nuclei",
                           "syllable_confidence": "estimated"})
    return result


def attach_performance(phrases, notes, tempo=90.0):
    """Attach expressive note events without changing the syllable targets."""
    beat = 60 / max(float(tempo), 40)
    for phrase in phrases:
        start, end = phrase["start_time"], phrase["end_time"]
        relevant = [dict(n) for n in notes if n.get("pitch_midi", 0) > 0
                    and n["start_time"] < end and n["end_time"] > start]
        phrase["note_events"] = relevant
        phrase["note_count"] = len(relevant)
        for slot in phrase.get("syllable_slots", []):
            slot["note_count"] = int(sum(n["start_time"] < slot["end_time"]
                                     and n["end_time"] > slot["start_time"] for n in relevant))
            slot["held"] = bool(slot["end_time"] - slot["start_time"] >= max(.42, beat * .7))
        target = int(phrase.get("target_syllables", phrase["syllables"]))
        phrase["max_words"] = target
        phrase["max_syllables"] = target + (0 if target <= 2 else 1)
        phrase["is_sustained"] = any(s.get("held") for s in phrase.get("syllable_slots", []))
        phrase["beat_position"] = round(start / beat, 2)
        pitches = [n["pitch_midi"] for n in relevant]
        phrase["note_pitch_range"] = max(pitches) - min(pitches) if pitches else 0
    return phrases
