import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Biases the model toward recognizing melodic/rap vocal content.
WHISPER_PROMPT = (
    "hip hop rap lyrics mumble singing vocal melody hook verse chorus "
    "yeah ayy ay aye ooh oh woah nah gonna wanna tryna gotta "
    "mm hmm ah la da ba bo doo woo hey yo na na "
    "singing humming melody vocal run riff ad-lib"
)

# Primary model — gpt-4o-transcribe is trained on far more diverse audio
# including singing, humming, and non-speech vocals. Better than whisper-1
# for melody captures, noise robustness, and varied cadence.
_PRIMARY_MODEL   = "gpt-4o-transcribe"
_FALLBACK_MODEL  = "whisper-1"


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using OpenAI's gpt-4o-transcribe (falls back to whisper-1).
    Returns text, word-level timestamps, and audio duration.
    """
    result = _transcribe_with(audio_path, _PRIMARY_MODEL)
    if result is None:
        result = _transcribe_with(audio_path, _FALLBACK_MODEL)
    if result is None:
        return {"text": "", "words": [], "duration": None}
    return result


def _transcribe_with(audio_path: str, model: str) -> dict | None:
    try:
        with open(audio_path, "rb") as f:
            kwargs = dict(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="en",
                prompt=WHISPER_PROMPT,
            )
            # whisper-1 accepts temperature; gpt-4o-transcribe ignores it but
            # passing it causes no error — leave it in for both models.
            if model == _FALLBACK_MODEL:
                kwargs["temperature"] = 0.1

            response = client.audio.transcriptions.create(**kwargs)

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
            "duration": getattr(response, "duration", None),
        }

    except Exception:
        return None
