# Lyric Goat — Mumble to Lyrics AI Engine

Record or upload a hum, mumble, gibberish topline or rough sung idea. The app
automatically finds vocal phrases, listens to the actual performance, and writes
three alternatives in your selected musical style. No song brief or timing
confirmation is required. Inputs are limited to 90-second sections.

## How It Works

1. Record or upload your vocal. Melody First is on by default.
2. Optionally choose genre, tone and section type.
3. Generate. Replay the original vocal alongside each result, or loop one phrase.
4. Keep or lock lines you like. Rewrite a line or request fresh versions.

The audio passes through separate time-aligned speech and melody branches.
Local analysis detects vocal activity and amplitude nuclei; pYIN (or optional
Basic Pitch) describes pitch events. A held syllable can span many notes, and
several syllables can share the same pitch. Notes never become syllable counts.

An audio-capable OpenAI model listens to the actual melody WAV for phonetic
hints, ending vowels and emotional delivery. High-confidence articulation counts
can resolve legato mumbles missed by amplitude analysis. Phrase boundaries stay
measured; any new internal subdivisions are explicitly marked as estimates.
Claude receives those performance constraints and creates three lyric versions.
Pronunciation-based checks flag poor syllable fit, overflow and stock phrases;
one optional batched repair edits only failed lines. Locked lines are preserved.

Timing and slot editing remain optional fine-tuning tools after generation.
Regenerating uses the existing analysis rather than uploading or listening again.

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
- **OpenAI key** → https://platform.openai.com (for native audio listening and optional transcription)

The app does not call either provider at startup. The offline detector benchmark
also makes no paid API calls.

### Models and call budget

- `OPENAI_LISTENING_MODEL=gpt-audio-1.5` receives the actual audio and returns text
  observations. `LYRIC_GOAT_AUDIO_LISTENER=auto` enables it when the OpenAI key is
  configured. Set it to `off` for local rhythm/pitch analysis only.
- A capture normally makes one listening call and one Claude writing call. A
  second Claude call occurs only when local fit/stock-phrase checks find failures.
- New Versions does not call the audio listener again. One-line rewrites use at
  most two writing calls and leave the old line intact if neither fits.
- Melody First skips speech transcription. With it off,
  `LYRIC_GOAT_TRANSCRIPTION_MODE=timed|semantic|hybrid` adds respectively one
  Whisper call, one GPT-4o Transcribe call, or both. Transcription failure can fall
  back to audio-only analysis; the response surfaces this state.
- Listener failures fall back to local timing and melody analysis with a visible
  status. This fallback cannot infer mouth sounds or delivery with the native
  audio model's detail. A Claude key is still required to write lyrics.
- Generation and listening are paid API usage, not included in ChatGPT. No model
  calls occur at startup or in the offline test suite/benchmark.

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
  analyze.py            Beat/pitch metadata and automatic vocal analysis
  performance.py        Phrase boundaries, syllable nuclei and multi-note runs
  listen.py             Actual-audio phonetic/feel listening with visible fallback
  lyric_craft.py        CMUdict syllables, stress, rhyme and stock-phrase checks
  melody_providers.py   pYIN plus optional local Basic Pitch adapter
  generate.py           Style-consistent writing, fit validation and bounded repair
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

## Verification

```bash
python -m unittest discover -s tests -v
# Optional Node >=22.12 frontend regression checks:
npm ci
npm test
```

Tests cover held runs, repeated syllables on one pitch, silence and original time
origin, actual-audio request construction, listener failure/count-confidence
handling, pronunciation mistakes, bounded lyric repair, locks, and a real WAV
upload through the pipeline with mocked external model responses. Frontend checks
cover current Copy/Save state, lock/context regeneration and silent playback gaps.
Synthetic signals and mocked providers are not a human vocal benchmark or proof
of lyric quality. Validate paid model calls and singability on real personal
recordings before relying on the output. A fit score measures programmed
constraints; it is not an artistic quality rating or a claim of Suno parity.

## Runtime and data

ffmpeg is required for compressed mic uploads. Gunicorn's 240-second request
budget accommodates the listening/writing stages. BPM is estimated unless the
user sets it or uses the metronome; a dry vocal does not establish the backing
track's true tempo, downbeat or absolute bar number. Timing is relative to the
original recording, so preserve your DAW export offset when working on a section.

Songs, audio and accepted-line feedback remain local files. Use a persistent
volume or local deployment to retain them. Feedback is collected but not yet used
as a trained writing profile. The server has no user authentication; keep access
private when storing unreleased recordings. Supabase is an inactive future schema.

## Next validation

Use the web UI with a 10-30 second take: one held run, a mumbled phrase with
consonants/vowels, then repeated syllables on a single note. Compare the words
against the original delivery, not just the fit score. Build the private
20-30-clip benchmark before choosing a different pitch provider. Artist-style
retrieval, hosted storage/auth and a Logic/AU plugin remain later work.

## Technical references

- [OpenAI audio input](https://developers.openai.com/api/docs/guides/audio-chat-completions)
- [CMU pronunciation dictionary](https://github.com/cmusphinx/cmudict)
- [REFFLY: melody-constrained lyric editing](https://aclanthology.org/2025.naacl-long.564/)
- [XAI-Lyricist: prosody-conditioned lyrics](https://github.com/lqhac/XAI-Lyricist)
