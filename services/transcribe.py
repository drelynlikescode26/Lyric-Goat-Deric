import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio file using OpenAI Whisper.
    Returns rough text + word-level timestamps.
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
        )

    words = []
    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
            })

    return {
        "text": response.text,
        "words": words,
        "duration": response.duration if hasattr(response, "duration") else None,
    }
