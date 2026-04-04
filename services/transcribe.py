import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Biases Whisper toward recognizing melodic/rap vocal content.
# Long list of filler sounds so Whisper maps "mmm" / "ayy" / "ooh"
# to real tokens rather than hallucinating random words.
WHISPER_PROMPT = (
    "hip hop rap lyrics mumble singing vocal melody hook verse chorus "
    "yeah ayy ay aye ooh oh woah nah gonna wanna tryna gotta "
    "mm hmm ah la da ba bo doo woo hey yo na na "
    "singing humming melody vocal run riff ad-lib"
)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper with music-optimized settings.
    Returns text, word-level timestamps, and audio duration.
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
            prompt=WHISPER_PROMPT,
            temperature=0.1,  # slight randomness helps with melodic mumbles vs. strict 0
        )

    words = []
    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word":  w.word,
                "start": w.start,
                "end":   w.end,
            })

    return {
        "text":     response.text.strip(),
        "words":    words,
        "duration": response.duration if hasattr(response, "duration") else None,
    }
