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
The lyric writer receives those performance constraints and creates three lyric versions.
Pronunciation-based checks flag poor syllable fit, overflow and stock phrases;
one optional batched repair edits only failed lines. Locked lines are preserved.

Timing and slot editing remain optional fine-tuning tools after generation.
Regenerating uses the existing analysis rather than uploading or listening again.

## Mac mini setup

The local version opens in Safari at **http://127.0.0.1:8765** while its launcher
is running. This address works only on the computer running the app; it is not a
hosted preview. The Mac installation must be completed before that link works.

1. Install [Homebrew](https://brew.sh) once if it is not already available.
2. Download this repository, then open `Install-Lyric-Goat.command`. If macOS
   asks, use Finder's Open action for the downloaded command file.
3. Setup installs Python 3.12 and ffmpeg through Homebrew, creates an isolated
   Python environment, and puts `Lyric-Goat.command` on your Desktop. The first
   run can take several minutes.
4. Safari opens the app's local OpenAI settings. Save your OpenAI API key there.
   A key is stored only in the installation's gitignored `.env`, readable by
   your Mac user. A saved key is never rendered back into the page.
5. Upload a 10–30 second vocal or allow Safari microphone access and record one.

The installed app lives in `~/Applications/Lyric-Goat`. Keep its Terminal window
open while using it. Close that window (or press Control-C) when finished.
Reopen the Desktop launcher for the next test: it checks GitHub `main` for
updates and installs changed dependencies. If an earlier instance is still
running, it reopens that instance; close its Terminal window before relaunching
to load updates. Modified tracked files, other branches and diverged history are
preserved rather than overwritten. Without GitHub access the installed version
still starts; generating lyrics requires an internet connection to OpenAI.

No GitHub login is needed to clone this public repository. Signing into OpenAI
in Safari alone does not configure the app; its API key is separate from your
ChatGPT subscription. Saving settings makes no API call and does not verify
billing, model access or subjective lyric quality. The first generated take
checks the real pipeline. Do not paste your key into chat or commit it to git.

### Models and cost

Only **one OpenAI API key** is required. The default writer is
`OPENAI_GENERATION_MODEL=gpt-6-luna`, with low reasoning, a 4,096 output-token
cap and no automatic SDK retries. The local Settings page also offers
`gpt-5.4-mini` for comparing writing on the same takes. No automatic fallback
switches you to a more expensive model. These are starting choices for real
vocal testing, not a claim that either reproduces your writing style.

Official standard text prices checked October 23, 2026:

| Lyric writer | Input / million tokens | Output / million tokens |
| --- | ---: | ---: |
| GPT-6 Luna (default) | $0.10 | $0.50 |
| GPT-5.4 Mini (optional comparison) | $0.75 | $4.50 |

For example, 6,000 input and 4,000 output tokens on Luna cost about **$0.0026**
for the writing call. Reasoning counts toward billed output. This is an example,
not a measured per-take price or spending cap; audio listening is additional.
See [OpenAI pricing](https://developers.openai.com/api/docs/pricing) and
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

- `OPENAI_LISTENING_MODEL=gpt-audio-1.5` hears the actual take and returns only
  text observations. Its audio input costs $32/million audio tokens, and text
  input/output costs $2.50/$10 per million. The app does not generate paid audio
  output. Audio tokens differ from text tokens. The older cheaper Audio Mini is
  deprecated, so it is not the default for a new installation.
- A capture normally makes one listening call and one writing call. A second
  writing call happens only when fit/stock-phrase checks find failed lines.
- New Versions reuses the listening analysis. One-line rewrites use at most two
  writing calls. All captures are limited to 90 seconds; start with short takes.
- Melody First skips separate speech transcription. Turning it off can add
  transcription charges. Native-listener failures visibly fall back to local
  rhythm/pitch analysis, which has less phonetic/feel detail.
- `LYRIC_GOAT_AUDIO_LISTENER=off` disables paid native listening, with that same
  tradeoff. Lyric writing still uses the selected paid writer.
- Nothing calls a paid model at startup or in offline tests/benchmarks.

Audio model references: [GPT Audio 1.5](https://developers.openai.com/api/docs/models/gpt-audio-1.5),
[Audio Mini status](https://developers.openai.com/api/docs/models/gpt-audio-mini).

### Existing installations and other platforms

Python 3.10–3.13 and ffmpeg are required. On Ubuntu/Debian, install ffmpeg with
`sudo apt install ffmpeg`. Then:

```bash
git clone https://github.com/drelynlikescode26/Lyric-Goat-Deric.git
cd Lyric-Goat-Deric
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_local.py
```

The local runner enables the key setup page, restricts access to loopback and
uses Waitress without a debugger. To choose a different port, set
`LYRIC_GOAT_PORT=8766` in `.env`. It detects another instance or port collision.
The root `python app.py` command also binds only to loopback, without key setup.

Existing Anthropic users can keep `.env` with `ANTHROPIC_API_KEY` and
`LYRIC_GOAT_WRITER=anthropic`; `ANTHROPIC_GENERATION_MODEL` remains configurable.
Without an explicit provider, `auto` prefers an available OpenAI key and otherwise
uses Anthropic. The Mac Settings page explicitly selects OpenAI. Hosted deployments
must set keys through their host's environment; local key setup is disabled there.

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
scripts/                Mac installer/launcher, private local runner and benchmark
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

ffmpeg is required for compressed mic uploads. The local runner binds only to 127.0.0.1. Gunicorn's 240-second request
budget accommodates the listening/writing stages. BPM is estimated unless the
user sets it or uses the metronome; a dry vocal does not establish the backing
track's true tempo, downbeat or absolute bar number. Timing is relative to the
original recording, so preserve your DAW export offset when working on a section.

Saved songs, audio and accepted-line feedback remain local files. Generating sends the take to OpenAI for listening and performance descriptions to the selected writer. Use a persistent
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
