import re
from collections import Counter
import numpy as np
import librosa

MAX_SYLLABLES_PER_BAR = 10

CHROMATIC_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Vowel family triggers — words/sounds that indicate dominant vowel shape
VOWEL_FAMILIES = {
    "ayy": ["ayy", "ay", "aye", "way", "say", "hey", "they", "yeah", "make", "take", "day"],
    "oh":  ["oh", "ooh", "woah", "no", "so", "go", "though", "low", "hold", "cold", "soul"],
    "ee":  ["ee", "free", "me", "be", "see", "feel", "need", "real", "deep", "sleep", "breathe"],
    "ah":  ["ah", "yah", "nah", "ma", "la", "aah", "heart", "dark", "hard", "far", "star"],
    "uh":  ["uh", "huh", "up", "love", "blood", "rough", "stuck", "cut", "run", "done"],
    "ii":  ["ii", "lie", "cry", "fly", "try", "die", "high", "mind", "time", "right", "night"],
}


def analyze_flow(audio_path: str, word_timestamps: list) -> dict:
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.squeeze(tempo))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    avg_energy = float(np.mean(rms))
    max_energy = float(np.max(rms))
    energy_ratio = avg_energy / max_energy if max_energy > 0 else 0

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(spectral_centroid))

    # Detect musical key via chromagram
    detected_key = _detect_key(y, sr)

    # Detect vowel patterns and repetition from word timestamps
    all_words = [w["word"] for w in word_timestamps]
    vowel_data = _detect_vowel_patterns(all_words)

    # Phrase detection + bar length enforcement
    raw_phrases = _detect_phrases(word_timestamps, gap_threshold=0.4)
    phrases = _split_long_phrases(raw_phrases, max_syllables=MAX_SYLLABLES_PER_BAR)

    if phrases:
        phrase_map = _build_phrase_map(phrases, beat_times)
    else:
        phrase_map = _build_melody_phrase_map(onset_times, beat_times)

    duration = round(float(len(y) / sr), 2)

    if phrase_map and phrase_map[-1].get("end_time", 0) <= phrase_map[-1]["start_time"]:
        phrase_map[-1]["end_time"] = duration

    flow_map = _build_flow_map(word_timestamps, beat_times)
    flow_style = _classify_flow(tempo, energy_ratio, avg_centroid, word_timestamps)

    # Use melody mode if very few real words OR transcript is just repetitions
    melody_mode = len(word_timestamps) < 3 or vowel_data["is_repetitive"]

    return {
        "tempo_bpm": tempo,
        "beat_count": len(beat_times),
        "syllable_count": len(onset_times),
        "energy_ratio": round(energy_ratio, 3),
        "flow_style": flow_style,
        "flow_map": flow_map,
        "phrase_map": phrase_map,
        "melody_mode": melody_mode,
        "duration": duration,
        "avg_words_per_beat": _words_per_beat(word_timestamps, beat_times),
        "detected_key": detected_key,
        "vowel_family": vowel_data["vowel_family"],
        "is_repetitive": vowel_data["is_repetitive"],
    }


def _detect_key(y: np.ndarray, sr: int) -> str:
    """Detect musical key using chromagram — most prominent pitch class."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = int(np.argmax(chroma_mean))
        return CHROMATIC_KEYS[key_idx]
    except Exception:
        return "C"


def _detect_vowel_patterns(words: list) -> dict:
    """
    Analyze transcript words for dominant vowel family and repetition.
    Returns vowel_family (str|None) and is_repetitive (bool).
    """
    if not words:
        return {"vowel_family": None, "is_repetitive": False}

    word_list = [w.lower().strip(".,!?") for w in words]

    # Repetition: most common word is >40% of total
    counts = Counter(word_list)
    most_common_count = counts.most_common(1)[0][1]
    is_repetitive = most_common_count / len(word_list) > 0.4

    # Find dominant vowel family
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


def _detect_phrases(word_timestamps: list, gap_threshold: float = 0.4) -> list:
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
            "text": text,
            "words": words,
            "syllables": syllables,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "beat_index": beat_idx,
        })

    return phrase_map


def _build_melody_phrase_map(onset_times: np.ndarray, beat_times: np.ndarray) -> list:
    if len(onset_times) == 0:
        return []

    phrases, current = [], [float(onset_times[0])]
    for i in range(1, len(onset_times)):
        if float(onset_times[i]) - float(onset_times[i - 1]) > 0.5:
            phrases.append(current)
            current = [float(onset_times[i])]
        else:
            current.append(float(onset_times[i]))
    if current:
        phrases.append(current)

    capped = []
    for phrase in phrases:
        while len(phrase) > MAX_SYLLABLES_PER_BAR:
            capped.append(phrase[:MAX_SYLLABLES_PER_BAR])
            phrase = phrase[MAX_SYLLABLES_PER_BAR:]
        if phrase:
            capped.append(phrase)

    phrase_map = []
    for phrase in capped:
        start_time = phrase[0]
        end_time = phrase[-1]
        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - start_time)))
        phrase_map.append({
            "text": "", "words": [],
            "syllables": len(phrase),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "beat_index": beat_idx,
        })

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
            "word": w["word"],
            "time": word_time,
            "beat_index": nearest_beat_idx,
            "beat_offset": round(beat_offset, 3),
            "duration": w.get("end", word_time) - word_time,
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
