import re
import numpy as np
import librosa

# Max syllables per bar — phrases longer than this get split at word boundaries
MAX_SYLLABLES_PER_BAR = 10


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

    # Detect phrases by silence gaps, then enforce bar length limit
    raw_phrases = _detect_phrases(word_timestamps, gap_threshold=0.4)
    phrases = _split_long_phrases(raw_phrases, max_syllables=MAX_SYLLABLES_PER_BAR)

    if phrases:
        phrase_map = _build_phrase_map(phrases, beat_times)
    else:
        phrase_map = _build_melody_phrase_map(onset_times, beat_times)

    # Add end_time to each phrase (used for word-level karaoke interpolation)
    duration = round(float(len(y) / sr), 2)
    for i, phrase in enumerate(phrase_map):
        if i + 1 < len(phrase_map):
            phrase["end_time"] = phrase_map[i + 1]["start_time"]
        else:
            phrase["end_time"] = duration

    flow_map = _build_flow_map(word_timestamps, beat_times)
    flow_style = _classify_flow(tempo, energy_ratio, avg_centroid, word_timestamps)
    melody_mode = len(word_timestamps) < 3

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
    """
    If a phrase has more syllables than max_syllables, split it at word
    boundaries so no bar is too long. Preserves original word timestamps.
    """
    result = []
    for phrase in phrases:
        total_syls = sum(_count_syllables(w["word"]) for w in phrase)
        if total_syls <= max_syllables:
            result.append(phrase)
            continue

        # Split into sub-bars at word boundaries
        current_bar = []
        current_syls = 0
        for word in phrase:
            w_syls = _count_syllables(word["word"])
            if current_syls + w_syls > max_syllables and current_bar:
                result.append(current_bar)
                current_bar = [word]
                current_syls = w_syls
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

    phrases = []
    current = [float(onset_times[0])]
    for i in range(1, len(onset_times)):
        gap = float(onset_times[i]) - float(onset_times[i - 1])
        if gap > 0.5:
            phrases.append(current)
            current = [float(onset_times[i])]
        else:
            current.append(float(onset_times[i]))
    if current:
        phrases.append(current)

    # Also enforce bar length on melody phrases (cap at MAX_SYLLABLES_PER_BAR onsets)
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
            "text": "",
            "words": [],
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
        bi = entry["beat_index"]
        beats.setdefault(bi, []).append(entry["word"])
    parts = []
    for bi in sorted(beats.keys()):
        parts.append(f"[beat {bi + 1}: {' '.join(beats[bi])}]")
    return " ".join(parts)
