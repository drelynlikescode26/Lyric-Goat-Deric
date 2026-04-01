# Lyric Goat — Mumble to Lyrics AI Engine

Turn your mumbles into polished lyrics. Record or upload a raw vocal idea, and the engine transcribes it, analyzes your flow, and generates multiple lyric versions matched to your cadence.

## How It Works

1. **Record** your mumble or **upload** an audio file
2. Set your **tone**, **mode**, and **vibe**
3. Hit **Generate** — get 3 lyric versions ranked by flow match

**Pipeline:**
```
Audio → Preprocess (normalize/resample) → Whisper (transcribe) → librosa (flow analysis) → Claude (3 lyric versions)
```

## Setup

### Requirements

- Python 3.10+
- **ffmpeg** (required for audio format conversion)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Install

```bash
git clone https://github.com/drelynlikescode26/Lyric-Goat-Deric
cd Lyric-Goat-Deric

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

### API Keys

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FLASK_SECRET_KEY=any-random-string
```

- **Anthropic key** → https://console.anthropic.com
- **OpenAI key** → https://platform.openai.com (for Whisper transcription)

### Run

```bash
python app.py
```

Open http://localhost:5000

## Style Controls

| Control | Options |
|---------|---------|
| **Tone** | Melodic · Aggressive · Simple · Punchlines |
| **Mode** | Verse · Hook · Story |
| **Vibe** | Introspective · Sad · Hype · Love |

## Project Structure

```
app.py                  Flask server + /process endpoint
services/
  preprocess.py         pydub: normalize + resample to 16kHz mono WAV
  transcribe.py         OpenAI Whisper: text + word timestamps
  analyze.py            librosa: tempo, beats, syllables, flow classification
  generate.py           Claude: 3 lyric versions, auto-scored and ranked
templates/index.html    UI
static/style.css        Styles
static/script.js        Mic recording, file upload, results rendering
```

## Roadmap

- **V1** (done) — upload/record → transcribe → generate
- **V2** (done) — syllable matching, flow analysis, multiple outputs, style controls
- **V3** — phoneme-level analysis, flow locking, style presets trained on your catalog
- **V4** — Logic Pro plugin, real-time suggestions, melody-aware writing
