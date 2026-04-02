import os
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_leading_silence


def preprocess(input_path: str) -> str:
    """
    Normalize, resample, and strip silence from audio.
    Returns path to a clean 16kHz mono WAV.
    Required: ffmpeg installed on the system.
    """
    audio = AudioSegment.from_file(input_path)

    # Convert to mono 16kHz (what Whisper + librosa expect)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Normalize volume
    audio = normalize(audio)

    # Strip leading/trailing silence (threshold -50dBFS)
    start_trim = detect_leading_silence(audio, silence_threshold=-50, chunk_size=10)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=-50, chunk_size=10)
    duration = len(audio)
    if start_trim + end_trim < duration:
        audio = audio[start_trim : duration - end_trim]

    output_path = os.path.splitext(input_path)[0] + "_clean.wav"
    audio.export(output_path, format="wav")
    return output_path
