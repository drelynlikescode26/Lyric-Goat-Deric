"""Create separate, timing-aligned audio branches for speech and melody."""

from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize


def preprocess_branches(input_path: str) -> dict:
    """Return speech and melody WAVs without changing their time origin.

    Speech: 16 kHz mono, normalized, pre-emphasized for consonants.
    Melody: 22.05 kHz mono, normalized, no speech-specific filtering.
    """
    with open(input_path, "rb") as input_file:
        source = normalize(AudioSegment.from_file(input_file).set_channels(1))
    stem = os.path.splitext(input_path)[0]
    speech_base_path = stem + "_speech_base.wav"
    speech_path = stem + "_speech.wav"
    melody_path = stem + "_melody.wav"

    speech_export = source.set_frame_rate(16000).export(speech_base_path, format="wav")
    melody_export = source.set_frame_rate(22050).export(melody_path, format="wav")
    speech_export.close()
    melody_export.close()

    try:
        speech, speech_sr = librosa.load(speech_base_path, sr=16000, mono=True)
        emphasized = _peak_normalize(librosa.effects.preemphasis(speech, coef=0.97))
        sf.write(speech_path, emphasized, speech_sr, subtype="PCM_16")
    finally:
        if os.path.exists(speech_base_path):
            os.unlink(speech_base_path)

    return {
        "speech_path": speech_path,
        "melody_path": melody_path,
        "timing_offset": 0.0,
    }


def preprocess(input_path: str) -> str:
    """Backward-compatible helper returning the speech branch only."""
    branches = preprocess_branches(input_path)
    melody_path = branches["melody_path"]
    if os.path.exists(melody_path):
        os.unlink(melody_path)
    return branches["speech_path"]


def _peak_normalize(samples: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return samples / peak * target if peak > 0 else samples
