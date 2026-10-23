# Lyric Goat — Mumble to Lyrics AI Engine

Turn your mumbles into polished lyrics. Record or upload a raw vocal idea, and the engine transcribes it, analyzes your flow, and generates multiple lyric versions matched to your cadence.

## How It Works

1. **Record** your mumble or **upload** an audio file
2. Set your **tone**, **mode**, and **vibe**
3. Hit **Generate** — get 3 lyric versions ranked by flow match

**Pipeline:**

```text
Audio ─┬─ speech branch (16 kHz) → timed/semantic transcription
       └─ melody branch (22.05 kHz) → phrasing, pitch, syllable slots
                            ↓
              reviewed slots → one structured Claude request → 3 ranked versions
```

The branches remain time-aligned. Editing a syllable slot or requesting new
versions does not upload or transcribe the audio again.

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

The app does not call either provider at startup. The offline detector benchmark
also makes no paid API calls.

### Cost modes

`LYRIC_GOAT_TRANSCRIPTION_MODE` controls transcription:

| Mode | Calls | Behavior |
|---|---:|---|
| `timed` | 1 | Default; Whisper word timestamps for slot alignment |
| `semantic` | 1 | GPT-4o Transcribe text; cadence comes from local analysis |
| `hybrid` | 2 | Semantic text plus Whisper timing |

Provider failures are returned with diagnostics instead of silently switching
models. Hum mode skips transcription completely.

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
  preprocess.py         Separate time-aligned speech and melody WAVs
  transcribe.py         Explicit OpenAI transcription modes + diagnostics
  analyze.py            librosa: tempo, beats, syllables, flow classification
  melody_providers.py   pYIN plus optional local Basic Pitch adapter
  generate.py           One structured Claude call, locally scored and ranked
  projects.py           Local-first songs, sections, and audio
  writing_profile.py    Accepted/edited/rejected lyric feedback history
benchmarks/             Personal clip manifest (audio stays gitignored)
scripts/                Offline detection benchmark
supabase/               Optional future hosted schema; not active by default
tests/                  Offline unit and smoke tests
templates/index.html    UI
static/style.css        Styles
static/script.js        Mic recording, file upload, results rendering
```

## Detector benchmark (free/local)

Copy `benchmarks/dataset.example.json` to `benchmarks/dataset.local.json`, add
your own clips under `benchmarks/clips/`, and mark the expected phrase starts,
ends, and syllable counts. Then run:

```bash
python scripts/benchmark_detection.py benchmarks/dataset.local.json \
  --output benchmarks/report.local.json
```

Personal audio, labels, and reports are ignored by git. To compare Spotify's
free local Basic Pitch model, install `requirements-melody.txt` and run with
`MELODY_NOTE_PROVIDER=basic_pitch`.

## Roadmap

- **V1** (done) — upload/record → transcribe → generate
- **V2** (done) — syllable matching, flow analysis, multiple outputs, style controls
- **V3** (active) — real-clip benchmark, editable syllable slots, accepted-line profile
- **V4** — hosted Supabase storage/auth after the local workflow proves reliable
- **V5** — Logic/AU plugin only after the web app consistently produces usable lyrics
