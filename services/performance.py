"""Automatic vocal phrasing and syllable nuclei, independent of pitch changes.

An amplitude nucleus is an estimate, not a decoded word. Pitch events are kept
separately so a run can carry one syllable and repeated notes can carry many.
No timing approval step or model/network call is required here.
"""
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, binary_closing
from scipy.signal import find_peaks


def _cadence_changes(nuclei, step):
    """Conservative change points in articulations, never in pitch-note counts.

    Compare several intervals on each side. A one-off pickup or syncopation
    must not turn into a new phrase; straight and repeating swung cadences can.
    These are heuristic suggestions, not calibrated musical bar boundaries.
    """
    times = nuclei * step
    intervals = np.diff(times)
    candidates = []

    def repeat_error(values):
        errors = [np.mean(np.abs(values - np.mean(values)))]
        if len(values) >= 4:
            alternating = np.array([np.mean(values[j % 2::2]) for j in range(len(values))])
            errors.append(np.mean(np.abs(values - alternating)))
        return min(errors) / max(float(np.mean(values)), .01)

    for i in range(4, len(nuclei) - 3):
        # Exclude the interval crossing the proposed boundary: it can be a
        # pickup or a stretched ending between two otherwise stable cadences.
        left = intervals[max(0, i - 5):i - 1]
        right = intervals[i:i + 4]
        if min(sum(left), sum(right)) < .45:
            continue
        if max(repeat_error(left), repeat_error(right)) > .08:
            continue
        speed = abs(np.log2(np.mean(left) / np.mean(right)))
        shape = 0.
        if min(len(left), len(right)) >= 4:
            shape = float(np.max(np.abs(
                np.quantile(left / np.mean(left), [.25, .75])
                - np.quantile(right / np.mean(right), [.25, .75]))))
        score = max(speed / np.log2(1.6), shape / .35)
        if score >= 1:
            candidates.append((score, i))
    selected = []
    # Prefer the earlier edge of a near-tied plateau rather than moving a
    # switch later merely because the next window has tiny frame jitter.
    for _, i in sorted(candidates, key=lambda item: (-round(item[0], 1), item[1])):
        if all(abs(i - j) >= 4 and abs(times[i] - times[j]) >= .8 for j in selected):
            selected.append(i)
    return sorted(selected)


def _phrase_parts(nuclei, envelope, start, end, step, tempo):
    # Use low-energy points between articulations. Shared edges preserve the
    # continuous timeline instead of inventing a silence between lyric lines.
    bounds = [start]
    for left, right in zip(nuclei, nuclei[1:]):
        margin = max(1, int((right - left) * .15))
        lo, hi = int(left + margin), int(right - margin)
        valley = envelope[lo:hi + 1]
        options = np.flatnonzero(valley <= valley.min() + float(valley.max()) * .005) + lo
        bounds.append(int(options[np.argmin(abs(options - (left + right) / 2))]))
    bounds.append(end)
    cuts = {0: "vocal_start", **{i: "cadence_change" for i in _cadence_changes(nuclei, step)}}
    # Estimated BPM supplies a phrase-length guide, not a claimed downbeat.
    # Length alone can divide multiple articulations, never a single held run.
    target_span = max(2., min(4., 4 * 60 / max(float(tempo), 40)))
    anchors = sorted(cuts) + [len(nuclei)]
    for a, b in zip(anchors, anchors[1:]):
        while b - a > 16 or (bounds[b] - bounds[a]) * step > target_span * 1.25:
            choices = list(range(a + 1, min(a + 16, b - 1) + 1))
            if not choices:
                break
            target = min(bounds[a] + target_span / step, bounds[choices[-1]])
            cut = min(choices, key=lambda j: abs(bounds[j] - target)
                      + max(0, .45 / step - (bounds[b] - bounds[j])))
            cuts[cut] = "syllable_limit" if cut == a + 16 else "phrase_length"
            a = cut
    points = sorted(cuts) + [len(nuclei)]
    return [(a, b, bounds[a:b + 1], cuts[a]) for a, b in zip(points, points[1:])]


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
        for offset, last, bounds, source in _phrase_parts(nuclei, envelope, start, end, step, tempo):
            part = nuclei[offset:last]
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
                           "boundary_source": source if offset else ("pause" if result else "vocal_start"),
                           "boundary_estimated": bool(offset),
                           "continuous_from_previous": bool(offset),
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
