import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Musical context prompt — biases Whisper toward rap/singing vocabulary
# and away from misinterpreting melody as speech noise.
WHISPER_PROMPT = (
    "hip hop rap lyrics mumble singing melody hook verse chorus "
    "yeah ayy aye ooh nah gonna wanna tryna gotta"
)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper with music-optimized settings.
    - prompt: biases vocabulary toward rap/singing
    - temperature=0: deterministic, most accurate decode
    - word timestamps: needed for phrase detection and karaoke sync
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
            prompt=WHISPER_PROMPT,
            temperature=0,
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
        "text": response.text.strip(),
        "words": words,
        "duration": response.duration if hasattr(response, "duration") else None,
    }
