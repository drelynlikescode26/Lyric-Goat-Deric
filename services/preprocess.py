import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_leading_silence


def preprocess(input_path: str) -> str:
    """
    Clean and optimize audio for Whisper transcription + librosa analysis.

    Pipeline:
    1. Convert to mono 16kHz WAV (what Whisper + librosa expect)
    2. Normalize volume
    3. Strip leading/trailing silence aggressively
    4. Apply pre-emphasis filter to boost vocal frequencies
       (makes consonants and articulation clearer for Whisper)

    Requires: ffmpeg installed on the system.
    """
    # Step 1-3: pydub handles format conversion, normalization, silence trimming
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = normalize(audio)

    # Trim leading silence (threshold -45dBFS, chunk 10ms)
    start_trim = detect_leading_silence(audio, silence_threshold=-45, chunk_size=10)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=-45, chunk_size=10)
    duration = len(audio)
    if start_trim + end_trim < duration:
        audio = audio[start_trim: duration - end_trim]

    # Write intermediate WAV for librosa
    intermediate_path = os.path.splitext(input_path)[0] + "_norm.wav"
    audio.export(intermediate_path, format="wav")

    # Step 4: Pre-emphasis filter via librosa
    # Boosts high frequencies (consonants, articulation) — makes mumbles
    # clearer to Whisper without changing perceived volume
    y, sr = librosa.load(intermediate_path, sr=16000)
    y_emphasized = librosa.effects.preemphasis(y, coef=0.97)

    # Normalize again after pre-emphasis so levels stay consistent
    peak = np.max(np.abs(y_emphasized))
    if peak > 0:
        y_emphasized = y_emphasized / peak * 0.95

    output_path = os.path.splitext(input_path)[0] + "_clean.wav"
    sf.write(output_path, y_emphasized, sr, subtype="PCM_16")

    # Clean up intermediate file
    if os.path.exists(intermediate_path):
        os.unlink(intermediate_path)

    return output_path
