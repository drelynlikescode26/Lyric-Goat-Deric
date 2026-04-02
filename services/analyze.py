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

    # Estimate syllable count from onsets
    estimated_syllables = len(onset_times)

    # Build flow map from word timestamps
    flow_map = _build_flow_map(word_timestamps, beat_times)

    # Classify flow style
    flow_style = _classify_flow(tempo, energy_ratio, avg_centroid, word_timestamps)

    return {
        "tempo_bpm": tempo,
        "beat_count": len(beat_times),
        "syllable_count": estimated_syllables,
        "energy_ratio": round(energy_ratio, 3),
        "flow_style": flow_style,
        "flow_map": flow_map,
        "avg_words_per_beat": _words_per_beat(word_timestamps, beat_times),
    }


def _build_flow_map(word_timestamps: list, beat_times: np.ndarray) -> list:
    """Map each word to its nearest beat position."""
    if not word_timestamps or len(beat_times) == 0:
        return []

    flow_map = []
    for w in word_timestamps:
        word_time = w.get("start", 0)
        # Find nearest beat
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
    """Classify the overall flow style based on audio features."""
    # High tempo + high energy = aggressive/rap
    if tempo > 130 and energy_ratio > 0.5:
        return "rap"
    # Slower tempo + lower energy = melodic/sung
    if tempo < 100 and energy_ratio < 0.4:
        return "melodic"
    # Mid tempo, dense syllables = trap/rhythmic
    if 90 <= tempo <= 140 and len(word_timestamps) > 10:
        return "rhythmic"
    return "mixed"


def syllable_rhythm_string(flow_map: list) -> str:
    """
    Convert flow map into a readable rhythm string like:
    [beat1: word1 word2] [beat2: word3] ...
    Useful for feeding into the prompt.
    """
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
