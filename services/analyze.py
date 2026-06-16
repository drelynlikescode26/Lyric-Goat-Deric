import re
from collections import Counter
import numpy as np
import librosa

from services.phrase_features import extract_phrase_features, phrase_debug_summary

# Semitone bin width for note quantization (in Hz, approximate)
_NOTE_MERGE_SEMITONES = 1.5  # merge consecutive frames within ±1.5 semitones

MAX_SYLLABLES_PER_BAR = 10

CHROMATIC_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Kessler key profiles (12-element, starting at C)
_KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

# Vowel family triggers
VOWEL_FAMILIES = {
    "ayy": ["ayy", "ay", "aye", "way", "say", "hey", "they", "yeah", "make", "take", "day"],
    "oh":  ["oh", "ooh", "woah", "no", "so", "go", "though", "low", "hold", "cold", "soul"],
    "ee":  ["ee", "free", "me", "be", "see", "feel", "need", "real", "deep", "sleep", "breathe"],
    "ah":  ["ah", "yah", "nah", "ma", "la", "aah", "heart", "dark", "hard", "far", "star"],
    "uh":  ["uh", "huh", "up", "love", "blood", "rough", "stuck", "cut", "run", "done"],
    "ii":  ["ii", "lie", "cry", "fly", "try", "die", "high", "mind", "time", "right", "night"],
}


def preanalyze_audio(audio_path: str, manual_bpm: float = 0) -> dict:
    """
    Pure-audio analysis that does NOT need word timestamps.
    Run this in parallel with transcription to save wall-clock time.
    Returns a dict that analyze_flow() accepts as `pre_data`.
    """
    y, sr = librosa.load(audio_path, sr=None)

    if manual_bpm > 0:
        tempo = float(manual_bpm)
        beat_interval = 60.0 / tempo
        duration_pre = float(len(y) / sr)
        beat_times = np.arange(0, duration_pre, beat_interval)
    else:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo      = float(np.squeeze(tempo))
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units="frames",
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.07, wait=10
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    rms        = librosa.feature.rms(y=y)[0]
    avg_energy = float(np.mean(rms))
    max_energy = float(np.max(rms))
    energy_ratio = avg_energy / max_energy if max_energy > 0 else 0

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(spectral_centroid))

    detected_key = _detect_key_kk(y, sr)
    duration     = round(float(len(y) / sr), 2)

    return {
        "y":            y,
        "sr":           sr,
        "tempo":        tempo,
        "beat_times":   beat_times,
        "onset_times":  onset_times,
        "energy_ratio": energy_ratio,
        "avg_centroid": avg_centroid,
        "detected_key": detected_key,
        "duration":     duration,
    }


def analyze_flow(audio_path: str, word_timestamps: list, pre_data: dict | None = None, hum_mode: bool = False) -> dict:
    """
    Full flow analysis.  Pass `pre_data` from preanalyze_audio() to skip
    re-loading audio and re-running librosa (saves ~1-2s when parallelized).
    """
    if pre_data is not None:
        y            = pre_data["y"]
        sr           = pre_data["sr"]
        tempo        = pre_data["tempo"]
        beat_times   = pre_data["beat_times"]
        onset_times  = pre_data["onset_times"]
        energy_ratio = pre_data["energy_ratio"]
        avg_centroid = pre_data["avg_centroid"]
        detected_key = pre_data["detected_key"]
        duration     = pre_data["duration"]
    else:
        pre_data     = preanalyze_audio(audio_path)
        y            = pre_data["y"]
        sr           = pre_data["sr"]
        tempo        = pre_data["tempo"]
        beat_times   = pre_data["beat_times"]
        onset_times  = pre_data["onset_times"]
        energy_ratio = pre_data["energy_ratio"]
        avg_centroid = pre_data["avg_centroid"]
        detected_key = pre_data["detected_key"]
        duration     = pre_data["duration"]

    # Vowel patterns + repetition detection
    all_words = [w["word"] for w in word_timestamps]
    vowel_data = _detect_vowel_patterns(all_words)

    # Phrase detection — choose audio-based or transcript-based
    raw_phrases = _detect_phrases(word_timestamps, gap_threshold=0.35)
    phrases     = _split_long_phrases(raw_phrases, max_syllables=MAX_SYLLABLES_PER_BAR)

    # Decide whether transcript is usable.
    # Criteria: must have ≥3 words AND transcript captured ≥30% of expected syllables.
    enough_words = len(word_timestamps) >= 3
    total_transcript_syls = sum(
        sum(_count_syllables(w["word"]) for w in p) for p in phrases
    ) if phrases else 0
    onset_expected = len(onset_times)
    transcript_too_sparse = (
        onset_expected > 5 and total_transcript_syls < onset_expected * 0.30
    )

    if not hum_mode and phrases and enough_words and not transcript_too_sparse:
        phrase_map = _build_phrase_map(phrases, beat_times)
        melody_mode = vowel_data["is_repetitive"]
    else:
        # Pure audio path: hum mode OR transcript unusable
        phrase_map = _build_melody_phrase_map(y, sr, onset_times, beat_times, duration)
        melody_mode = True

    if phrase_map and phrase_map[-1].get("end_time", 0) <= phrase_map[-1]["start_time"]:
        phrase_map[-1]["end_time"] = duration

    # For transcript-based maps with low-confidence phrases,
    # correct syllable counts using audio energy peaks.
    if not melody_mode:
        phrase_map = _correct_syllables_from_audio(phrase_map, y, sr)

    # Enrich each phrase with full feature set
    phrase_map = extract_phrase_features(
        y, sr, phrase_map, onset_times, duration,
        global_vowel_family=vowel_data["vowel_family"],
    )

    # Build note map: discrete notes from pyin pitch + onset grid.
    # Attaches note_count to each phrase so the prompt knows exact syllable slots.
    note_map = _build_note_map(y, sr, onset_times, beat_times, duration)
    phrase_map = _attach_note_counts(phrase_map, note_map)

    flow_map = _build_flow_map(word_timestamps, beat_times)
    flow_style = _classify_flow(tempo, energy_ratio, avg_centroid, word_timestamps)

    # Debug summary for the UI audit panel
    debug_phrases = phrase_debug_summary(phrase_map)

    return {
        "tempo_bpm":         tempo,
        "beat_count":        len(beat_times),
        "syllable_count":    len(onset_times),
        "energy_ratio":      round(energy_ratio, 3),
        "flow_style":        flow_style,
        "flow_map":          flow_map,
        "phrase_map":        phrase_map,
        "note_map":          note_map,
        "melody_mode":       melody_mode,
        "duration":          duration,
        "avg_words_per_beat": _words_per_beat(word_timestamps, beat_times),
        "detected_key":      detected_key,
        "vowel_family":      vowel_data["vowel_family"],
        "is_repetitive":     vowel_data["is_repetitive"],
        "debug_phrases":     debug_phrases,
    }


def _detect_key_kk(y: np.ndarray, sr: int) -> str:
    """
    Detect musical key using Krumhansl-Kessler tonal hierarchy profiles.
    Compares chroma energy distribution against major + minor templates
    for all 12 roots, picks the best correlation.
    Returns e.g. "A minor" or "C# major".
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        # Normalize so it sums to 1
        chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-9)

        best_score = -np.inf
        best_key = "C major"

        for root in range(12):
            # Rotate profile to match root
            major_profile = np.roll(_KK_MAJOR, root)
            minor_profile = np.roll(_KK_MINOR, root)

            maj_score = float(np.corrcoef(chroma_norm, major_profile)[0, 1])
            min_score = float(np.corrcoef(chroma_norm, minor_profile)[0, 1])

            if maj_score > best_score:
                best_score = maj_score
                best_key = f"{CHROMATIC_KEYS[root]} major"
            if min_score > best_score:
                best_score = min_score
                best_key = f"{CHROMATIC_KEYS[root]} minor"

        return best_key
    except Exception:
        return "C major"


def _build_note_map(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    total_duration: float,
) -> list:
    """
    Convert raw onset + pyin pitch data into a list of discrete notes.

    Each note is a dict:
      start_time, end_time, duration, pitch_midi (0 if unvoiced),
      pitch_name (e.g. "A4"), beat_index

    This gives the prompt exact syllable-slot counts per phrase rather than
    loose caps, matching the syllable-per-note alignment model.
    """
    if len(onset_times) == 0:
        return []

    # pyin gives per-frame fundamental frequency + voiced boolean
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
    except Exception:
        f0 = np.zeros(len(y) // 512 + 1)
        voiced_flag = np.zeros_like(f0, dtype=bool)

    hop_len   = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_len)

    notes = []
    onset_arr = np.array([float(t) for t in onset_times])

    for i, t_start in enumerate(onset_arr):
        t_end = float(onset_arr[i + 1]) if i + 1 < len(onset_arr) else total_duration
        duration = t_end - t_start

        # Get pyin frames in this note window
        mask = (frame_times >= t_start) & (frame_times < t_end) & voiced_flag
        voiced_f0 = f0[mask]

        if len(voiced_f0) > 0 and np.any(voiced_f0 > 0):
            avg_hz = float(np.nanmedian(voiced_f0[voiced_f0 > 0]))
            pitch_midi = round(librosa.hz_to_midi(avg_hz))
            pitch_midi = int(np.clip(pitch_midi, 0, 127))
            pitch_name = librosa.midi_to_note(pitch_midi)
        else:
            pitch_midi = 0
            pitch_name = "—"

        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - t_start)))

        notes.append({
            "start_time":  round(t_start, 3),
            "end_time":    round(t_end, 3),
            "duration":    round(duration, 3),
            "pitch_midi":  pitch_midi,
            "pitch_name":  pitch_name,
            "beat_index":  beat_idx,
        })

    return notes


def _attach_note_counts(phrase_map: list, note_map: list) -> list:
    """
    For each phrase, count how many notes fall inside its time window and
    store as `note_count`. Also compute `note_pitch_range` (semitone span)
    and `avg_pitch_midi` for the phrase — used in prompt constraints.
    """
    if not note_map:
        return phrase_map

    for phrase in phrase_map:
        t0 = phrase.get("start_time", 0)
        t1 = phrase.get("end_time", t0 + 0.5)

        phrase_notes = [
            n for n in note_map
            if n["start_time"] >= t0 - 0.05 and n["start_time"] < t1 + 0.05
        ]

        voiced = [n["pitch_midi"] for n in phrase_notes if n["pitch_midi"] > 0]

        phrase["note_count"]      = len(phrase_notes)
        phrase["note_pitch_range"] = (max(voiced) - min(voiced)) if len(voiced) > 1 else 0
        phrase["avg_pitch_midi"]   = round(float(np.mean(voiced))) if voiced else 0

        # Use note_count as the authoritative syllable target when it's
        # more informative than the transcript syllable count.
        # Only override if note_count is within a 2x range of transcript syls
        # (prevents noisy onsets from blowing up the count).
        transcript_syls = phrase.get("syllables", 0)
        if len(phrase_notes) > 0:
            ratio = len(phrase_notes) / max(transcript_syls, 1)
            if 0.5 <= ratio <= 2.0:
                phrase["target_syllables"] = len(phrase_notes)
            else:
                phrase["target_syllables"] = transcript_syls
        else:
            phrase["target_syllables"] = transcript_syls

    return phrase_map


def _detect_vowel_patterns(words: list) -> dict:
    if not words:
        return {"vowel_family": None, "is_repetitive": False}

    word_list = [w.lower().strip(".,!?") for w in words]

    counts = Counter(word_list)
    most_common_count = counts.most_common(1)[0][1]
    is_repetitive = most_common_count / len(word_list) > 0.4

    all_text = " ".join(word_list)
    best_family = None
    best_score = 0
    for family, triggers in VOWEL_FAMILIES.items():
        score = sum(1 for t in triggers if t in all_text)
        if score > best_score:
            best_score = score
            best_family = family

    return {
        "vowel_family": best_family if best_score > 0 else None,
        "is_repetitive": is_repetitive,
    }


def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _detect_phrases(word_timestamps: list, gap_threshold: float = 0.35) -> list:
    if not word_timestamps:
        return []

    phrases = []
    current = [word_timestamps[0]]

    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1].get("end", word_timestamps[i - 1].get("start", 0))
        curr_start = word_timestamps[i].get("start", 0)
        if curr_start - prev_end > gap_threshold:
            phrases.append(current)
            current = [word_timestamps[i]]
        else:
            current.append(word_timestamps[i])

    if current:
        phrases.append(current)

    return phrases


def _split_long_phrases(phrases: list, max_syllables: int = MAX_SYLLABLES_PER_BAR) -> list:
    result = []
    for phrase in phrases:
        total_syls = sum(_count_syllables(w["word"]) for w in phrase)
        if total_syls <= max_syllables:
            result.append(phrase)
            continue

        current_bar, current_syls = [], 0
        for word in phrase:
            w_syls = _count_syllables(word["word"])
            if current_syls + w_syls > max_syllables and current_bar:
                result.append(current_bar)
                current_bar, current_syls = [word], w_syls
            else:
                current_bar.append(word)
                current_syls += w_syls
        if current_bar:
            result.append(current_bar)

    return result


def _build_phrase_map(phrases: list, beat_times: np.ndarray) -> list:
    phrase_map = []
    for phrase in phrases:
        words = [w["word"] for w in phrase]
        text = " ".join(words)
        syllables = sum(_count_syllables(w) for w in words)
        start_time = phrase[0].get("start", 0)
        end_time = phrase[-1].get("end", start_time)

        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - start_time)))

        phrase_map.append({
            "text":       text,
            "words":      words,
            "syllables":  syllables,
            "start_time": round(start_time, 2),
            "end_time":   round(end_time, 2),
            "beat_index": beat_idx,
        })

    return phrase_map


def _build_melody_phrase_map(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    total_duration: float,
) -> list:
    """
    Build phrase map from audio alone — used when transcript is unusable.

    Splits audio into phrase groups using onset gaps, then counts syllables
    using RMS energy peaks (much more accurate than raw onset count for melody).
    """
    if len(onset_times) == 0:
        return []

    # Group onsets into phrase windows by gaps > 0.5s
    groups, current = [], [float(onset_times[0])]
    for i in range(1, len(onset_times)):
        gap = float(onset_times[i]) - float(onset_times[i - 1])
        if gap > 0.5:
            groups.append(current)
            current = [float(onset_times[i])]
        else:
            current.append(float(onset_times[i]))
    if current:
        groups.append(current)

    # Cap groups at MAX_SYLLABLES_PER_BAR by splitting large ones
    capped = []
    for g in groups:
        while len(g) > MAX_SYLLABLES_PER_BAR:
            capped.append(g[:MAX_SYLLABLES_PER_BAR])
            g = g[MAX_SYLLABLES_PER_BAR:]
        if g:
            capped.append(g)

    phrase_map = []
    for gi, group in enumerate(capped):
        t_start = group[0]
        # Phrase ends at start of next group (or total duration)
        if gi + 1 < len(capped):
            t_end = capped[gi + 1][0] - 0.05
        else:
            t_end = total_duration
        t_end = max(t_end, t_start + 0.1)

        # Count syllables from RMS energy peaks — more accurate for melody
        syl_count = _count_syllables_from_audio(y, sr, t_start, t_end)

        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - t_start)))

        phrase_map.append({
            "text":       "",
            "words":      [],
            "syllables":  syl_count,
            "start_time": round(t_start, 2),
            "end_time":   round(t_end, 2),
            "beat_index": beat_idx,
        })

    return phrase_map


def _count_syllables_from_audio(y: np.ndarray, sr: int, t_start: float, t_end: float) -> int:
    """
    Count syllable-like energy peaks in a time window using RMS envelope.

    Uses short-time RMS (25ms frames, 10ms hop) and finds local maxima
    with minimum 80ms separation — the typical minimum syllable duration.
    Falls back to onset count if scipy is unavailable.
    """
    s = max(0, int(t_start * sr))
    e = min(len(y), int(t_end * sr))
    seg = y[s:e]
    if len(seg) < 100:
        return 1

    frame_len = max(64, int(0.025 * sr))
    hop_len   = max(32, int(0.010 * sr))
    rms = librosa.feature.rms(y=seg, frame_length=frame_len, hop_length=hop_len)[0]
    if len(rms) < 3:
        return 1

    rms_norm = rms / (float(np.max(rms)) + 1e-9)
    min_dist = max(1, int(0.08 / (hop_len / sr)))  # 80ms minimum syllable gap

    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms_norm, distance=min_dist, prominence=0.12, height=0.15)
        count = len(peaks)
    except ImportError:
        # Simple threshold crossing fallback
        threshold, count, above = 0.25, 0, False
        for val in rms_norm:
            if val > threshold and not above:
                count += 1
                above = True
            elif val <= threshold:
                above = False

    return max(1, min(count, MAX_SYLLABLES_PER_BAR))


def _correct_syllables_from_audio(phrase_map: list, y: np.ndarray, sr: int) -> list:
    """
    For transcript-based phrases where Whisper underestimated syllables
    (low confidence, very short word count vs duration), correct using
    audio energy peaks.
    """
    for phrase in phrase_map:
        conf = phrase.get("confidence_label", "med")
        transcript_syls = phrase.get("syllables", 1)
        t_start = phrase.get("start_time", 0)
        t_end   = phrase.get("end_time", t_start + 0.5)
        p_dur   = max(t_end - t_start, 0.1)

        # Only correct when transcript seems sparse relative to audio duration
        syls_per_sec = transcript_syls / p_dur
        should_correct = conf == "low" or (syls_per_sec < 1.0 and not phrase.get("is_sustained", False))

        if should_correct:
            audio_syls = _count_syllables_from_audio(y, sr, t_start, t_end)
            if audio_syls > transcript_syls:
                phrase["syllables"] = audio_syls
                phrase["syllable_source"] = "audio"

    return phrase_map


def _build_flow_map(word_timestamps: list, beat_times: np.ndarray) -> list:
    if not word_timestamps or len(beat_times) == 0:
        return []

    flow_map = []
    for w in word_timestamps:
        word_time = w.get("start", 0)
        nearest_beat_idx = int(np.argmin(np.abs(beat_times - word_time)))
        beat_offset = word_time - float(beat_times[nearest_beat_idx])
        flow_map.append({
            "word":       w["word"],
            "time":       word_time,
            "beat_index": nearest_beat_idx,
            "beat_offset": round(beat_offset, 3),
            "duration":   w.get("end", word_time) - word_time,
        })

    return flow_map


def _words_per_beat(word_timestamps: list, beat_times: np.ndarray) -> float:
    if len(beat_times) == 0 or not word_timestamps:
        return 0.0
    return round(len(word_timestamps) / len(beat_times), 2)


def _classify_flow(tempo, energy_ratio, spectral_centroid, word_timestamps):
    if tempo > 130 and energy_ratio > 0.5:
        return "rap"
    if tempo < 100 and energy_ratio < 0.4:
        return "melodic"
    if 90 <= tempo <= 140 and len(word_timestamps) > 10:
        return "rhythmic"
    return "mixed"


def syllable_rhythm_string(flow_map: list) -> str:
    if not flow_map:
        return ""
    beats: dict = {}
    for entry in flow_map:
        beats.setdefault(entry["beat_index"], []).append(entry["word"])
    return " ".join(f"[beat {bi+1}: {' '.join(beats[bi])}]" for bi in sorted(beats.keys()))
