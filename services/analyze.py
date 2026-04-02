import re
import numpy as np
import librosa


def analyze_flow(audio_path: str, word_timestamps: list) -> dict:
    """
    Analyze audio for rhythm, tempo, syllable timing, and flow pattern.
    Returns a flow template the lyric generator can use.
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Tempo and beat tracking
    # librosa >= 0.10 returns tempo as a 1-element array, so squeeze to scalar
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.squeeze(tempo))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Onset detection (where syllables/notes hit)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # RMS energy over time (dynamics/intensity)
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = float(np.mean(rms))
    max_energy = float(np.max(rms))
    energy_ratio = avg_energy / max_energy if max_energy > 0 else 0

    # Spectral features for tonal quality
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(spectral_centroid))

    # Detect natural phrase breaks from word timing gaps
    phrases = _detect_phrases(word_timestamps, gap_threshold=0.4)

    # Build phrase map with syllable counts per line
    # If no real words detected (melody-only mumble), fall back to onset-based phrases
    if phrases:
        phrase_map = _build_phrase_map(phrases, beat_times)
    else:
        phrase_map = _build_melody_phrase_map(onset_times, beat_times)

    # Build flow map from word timestamps
    flow_map = _build_flow_map(word_timestamps, beat_times)

    # Classify flow style
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
        "avg_words_per_beat": _words_per_beat(word_timestamps, beat_times),
    }


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a single word."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    # Silent trailing e (e.g. "make", "time")
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _detect_phrases(word_timestamps: list, gap_threshold: float = 0.4) -> list:
    """
    Split word timestamps into natural phrases based on pauses.
    A gap longer than gap_threshold seconds = new phrase (new lyric line).
    """
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


def _build_phrase_map(phrases: list, beat_times: np.ndarray) -> list:
    """
    Build a line-by-line template: original words, syllable count, beat position.
    This is the key structure the lyric generator uses to match lines.
    """
    phrase_map = []
    for phrase in phrases:
        words = [w["word"] for w in phrase]
        text = " ".join(words)
        syllables = sum(_count_syllables(w) for w in words)
        start_time = phrase[0].get("start", 0)

        # Find which beat this phrase starts on
        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - start_time)))

        phrase_map.append({
            "text": text,
            "words": words,
            "syllables": syllables,
            "start_time": round(start_time, 2),
            "beat_index": beat_idx,
        })

    return phrase_map


def _build_melody_phrase_map(onset_times: np.ndarray, beat_times: np.ndarray) -> list:
    """
    When no words are detected (pure melody/hum), build a phrase map from
    onset clustering. Groups onsets into phrases using silence gaps, then
    assigns a syllable count equal to the number of onsets per phrase.
    """
    if len(onset_times) == 0:
        return []

    # Group onsets into phrases by silence gaps > 0.5s
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

    phrase_map = []
    for phrase in phrases:
        syllable_count = len(phrase)
        start_time = phrase[0]
        beat_idx = 0
        if len(beat_times) > 0:
            beat_idx = int(np.argmin(np.abs(beat_times - start_time)))
        phrase_map.append({
            "text": "",           # no words — melody only
            "words": [],
            "syllables": syllable_count,
            "start_time": round(start_time, 2),
            "beat_index": beat_idx,
        })

    return phrase_map


def _build_flow_map(word_timestamps: list, beat_times: np.ndarray) -> list:
    """Map each word to its nearest beat position."""
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


def _classify_flow(
    tempo: float,
    energy_ratio: float,
    spectral_centroid: float,
    word_timestamps: list,
) -> str:
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
        words = " ".join(beats[bi])
        parts.append(f"[beat {bi + 1}: {words}]")

    return " ".join(parts)
