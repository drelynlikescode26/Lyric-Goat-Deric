# Lyric Goat — Mumble to Lyrics AI Engine

Record or upload a hum, mumble, gibberish topline or rough sung idea. The app
automatically finds vocal phrases, listens to the actual performance, and writes
three alternatives in your selected musical style. No song brief or timing
confirmation is required. Inputs are limited to 90-second sections.

## How It Works

1. Record or upload your vocal. Melody First is on by default.
2. Choose your genre, type a theme, and select Clean or Explicit allowed. Tone and section type remain optional.
3. Generate. Replay the original vocal alongside each result, or loop one phrase.
4. Keep or lock lines you like. Rewrite a line or request fresh versions.

The audio passes through separate time-aligned speech and melody branches.
Local analysis detects vocal activity and amplitude nuclei; pYIN (or optional
Basic Pitch) describes pitch events. A held syllable can span many notes, and
several syllables can share the same pitch. Notes never become syllable counts.

Continuous mumbles do not need a pause to start another lyric line. Sustained
changes in articulation speed or repeating rhythm can suggest a flow boundary.
Long connected passages also get shorter phrase windows, guided by roughly four
beats (a 2–4 second target, with some tolerance), plus the 16-syllable limit.
Breaks use low-energy points between detected articulations and share an edge,
so they do not add silence. A single held vowel or run stays intact. These are
estimated lyric phrase breaks, not confirmed bars/downbeats; missed legato
articulations and real vocal flow changes still need testing on personal takes.
Record or upload again after a detector change: New Versions reuses old analysis.

Detailed listening (GPT Audio 1.5) is the studio default, selected after the
artist reviewed the same Moonlight take with multiple listening approaches.
It describes the performed gibberish and proposes its own phrase windows without
requiring silence. Phrase boundaries and counts are AI estimates, not measured
bars or verified per-syllable timings. Unknown counts use a labeled acoustic
estimate. Listener failures visibly retain local acoustic analysis.

The typed theme supplies meaning; the performed sounds supply the pocket.
GPT-6 Luna writes three real-word alternatives using the phonetic template,
repetitions, delivery and ending vowels. Guessed listener words are excluded
from the default writing input. Counts must match exactly: pronunciation checks
flag both shortened and overfilled lines, and one bounded repair edits failed
lines. Any remaining mismatch is shown for review, not silently accepted.
Locked lines stay unchanged. Theme and listening context persist with a saved
section and carry into New Versions and one-line rewrites.

Clean is the default; explicit language requires the explicit choice. That choice
persists with a saved section and reaches initial generation, line rewrites and
sound-edit updates. Common profanity in Clean output is repaired or withheld;
locked explicit lines require the artist to unlock them or choose Explicit.

Rhyme intent follows repeated performed endings and optional internal sound
anchors, including returns on non-adjacent lines. The writer receives the full
take's tentative sound relationships during both full and one-line generation.
Unknown relationships are neutral; a repeated vowel alone is only an assonance
clue. Genre guidance does not impose a fixed rhyme scheme, new bar lengths,
triplets, dialect, or invented biography. Rhyme scoring compares linked lines
instead of rewarding every neighboring pair. Edited sounds invalidate stale
listener annotations. Corniness and narrative coherence still need human review;
these writing instructions and limited stock-phrase checks are not taste scores.

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

Only **one OpenAI API key** is required. The writer is fixed to `gpt-6-luna`,
with low reasoning, a 4,096 output-token cap and no automatic SDK retries.
There is no writing-model picker or automatic switch to a more expensive model.
Older `OPENAI_GENERATION_MODEL` overrides are ignored; optional legacy Anthropic
installations keep their separate explicit provider configuration.

Official standard text prices checked October 23, 2026:

| Lyric writer | Input / million tokens | Output / million tokens |
| --- | ---: | ---: |
| GPT-6 Luna (default) | $0.10 | $0.50 |

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

## Compare listeners on the same take (local Mac)

Open **Compare listeners** from the studio, or use **Compare this take** to carry
the current recording into the comparison page. The Mac installer also creates
**Compare-Lyric-Goat.command** on the Desktop. Start with a 10–30 second section;
the maximum is 90 seconds. Save the take once.
Select listeners and click **Run comparison**. Saving the take and opening past
results make no paid calls. Every selected cloud option can incur its own API
charges; comparison does not generate lyrics or change the studio's default.

- **OpenAI current** preserves the original listener inside acoustic phrase windows.
- **OpenAI detailed** is now the studio default. It uses the same audio model with an independent, more detailed
  task: proposed phrase boundaries, phonetic hints, counts, delivery, and words.
  These times/counts are AI estimates. It is a prompt comparison, not a new model.
- **GPT Transcribe** is a speech-text baseline using `gpt-transcribe`. Its transcript
  is shown without invented musical phrase timings.
- **Gemini detailed** uses the same detailed task through Google's API. Add a
  separate Google key in the private API settings page to enable it. The default
  is `gemini-3.8-flash`; model availability depends on the Google project.
- **VocalParse experimental** is an optional isolated subprocess adapter. It is
  unavailable until `VOCALPARSE_PYTHON` names a Python executable with the official
  [VocalParse](https://github.com/pymaster17/VocalParse) package installed, and
  `VOCALPARSE_CHECKPOINT` names its downloaded checkpoint directory. Set those in
  the installation's private `.env` and restart. Follow the project's installation
  instructions in that separate environment; do not install its PyTorch stack in
  Lyric Goat's `.venv`. The published loader currently chooses CUDA or CPU rather
  than Apple MPS. The published checkpoint is primarily trained on Mandarin singing,
  requires sections no longer than 30 seconds, and is not a validated English-mumble
  listener. CPU speed/memory use on an 8 GB Mac are unverified. No research
  weights or dependencies are downloaded automatically. A run is limited to 180
  seconds, with an explicit failure if it cannot finish. Its predicted notation
  is shown as evidence, without equating note counts to syllables.

All listeners receive the same normalized mono WAV, identified by SHA-256.
Loop any proposed phrase against that original audio. Rate flow, sounds, and
syllable fit from 1–5, note mistakes, and mark your preferred result. These are
your judgments, not machine accuracy scores or automatic model training.
Reopen the saved take and run again after tweaks: earlier runs, settings version,
ratings, and audio remain in gitignored `data/comparisons/`. Detector analysis is
refreshed when the listening setup changes. Failures never silently substitute
another model. An interrupted run can be rerun without uploading again.

The page, audio, and results are available only in the loopback local runner.
The compare API checks a session CSRF token. Keys, provider error bodies, and
audio payloads are not included in saved result JSON. No comparisons run on page
load or ordinary lyric generation.

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
# Private cloud library

The Mac studio can save its library to a dedicated private Cloudflare R2 bucket.
Open **Mac storage** in the studio, enter the account ID, bucket name, and the
Access Key ID / Secret Access Key from an R2 token scoped to **Object Read & Write**
on that bucket only. Keep public access disabled. The connection test uses a
disposable text object, verifies it, and removes that test object before enabling
cloud saves. Credentials stay in the ignored, permission-restricted `.env` file.

Saved song JSON, saved audio, listening comparisons, and writing feedback upload
through a durable background queue. Upload failures keep the originals and pending
snapshots on disk and retry. The next launch reconciles library files again.
Cloud confirmation checks SHA-256 metadata and size. Only verified, unchanged audio
can leave the 50 MB local audio cache; active playback/analysis and pending saves
are protected. **Free synced audio cache** releases eligible local audio; reopening
it downloads and verifies the cloud object. Temporary processing files still use
the Mac. Offline pending saves can exceed the cache target to preserve work.

Each uploaded metadata revision has a separate cloud history object. Deleting a
saved item adds a cloud tombstone and keeps its previous remote contents; a history
restore screen is not yet included. **Open cloud library** restores missing metadata
without overwriting local edits. This is a single-Mac cloud library, not concurrent
multi-device editing: the phone review site's browser-local drafts are not synced.
Cloud storage is not an independent backup; export important lyrics/audio too.
R2 activation, storage/operation limits, and billing are controlled by Cloudflare.

## Lyric Desk: text-only lyric editing

Open **Lyric Desk** from the studio header or **Revise existing lyrics** on the
start screen. Paste a full song, click a lyric line or select up to eight lines,
and ask for a rewrite, clearer wording, a rhyming line, or a next line. Three
options are previewed before anything changes. Try different options from the
same set without another AI request; Undo restores the previous words. Blank
lines and section headings stay outside selected rewrite requests.

The fixed GPT-6 Luna writer uses the full draft as context and follows genre,
theme, clean/explicit language, and optional syllable-length/rhyme guidance.
Counts are pronunciation estimates; the user should sing the result to check fit.
There is at most one repair request per suggestion set. No audio is collected
and no AI calls happen on typing, saving, opening, or applying an existing option.

Browser backups keep in-progress edits. Saved notebooks live separately in
`data/notebooks/` and use the private R2 outbox/history when configured. Concurrent
save revisions are checked; **Save a copy** preserves work when another tab saved
a newer version. Copy and plain-text export keep the whole lyric sheet together.
The phone comparison Site does not yet host this new Mac workspace.

Workflow references: LyricStudio's public editor preview at https://lyricstudio.net/
and its official guide at https://blog.lyricstudio.net/2021/02/14/lyricstudio-hidden-tricks/.
Implementation and visual identity are Lyric Goat's own.

### Writing workshop: locks, Rhyme Radar, versions and hooks

- **Protect selected words / line** locks highlighted words, or the current line
  when there is no highlighted text. Locked phrases keep their exact spelling,
  punctuation, case, order and line during AI rewrites. Fully locked selections
  are rejected before spending credit. A failed protection check gets at most
  one repair; an option that still violates locks is never offered. Insertions
  above a protected phrase move its lock. Direct typing inside a lock removes it
  with a visible notice; locks protect AI edits, not the artist's own keyboard.
- **Rhyme Radar** opens a colored sound map with internal/ending rhymes,
  multisyllable and cross-word clues, tentative shared-vowel slant rhymes, and
  estimated syllable counts. Repeated words are separate; unknown pronunciations
  are named, not guessed. Filters focus on multisyllable or possible slant clues.
  It uses the existing local CMUdict and makes no AI request. These are English
  pronunciation estimates, not a performance score or a forced rhyme scheme.
- **Versions** compares an earlier draft with current words. Recover one changed,
  added or deleted line, or restore the whole draft. Each AI audition and named
  restore has its own entry; ordinary writing saves within a minute are grouped.
  **Save milestone** preserves a checkpoint. Up to 60 snapshots live inside each
  notebook, retaining the first draft and latest 59. Locks travel with whole-draft
  restores. Every saved notebook, including its timeline, enters the private R2
  queue; existing R2 immutable upload history is separate from the 60-entry UI.
- **Hook Workshop** works with a whole song or a theme/central phrase on a blank
  page. Request two, four or eight lines and audition three approaches: close and
  personal, a repeated refrain, or setup and payoff. The central phrase is kept
  exactly when supplied. Add a new `[Hook]` at the end, or replace a selection
  of the matching length. The full song stays in context; verses remain intact.
  Hook length matching is optional and defaults off for new hooks. Auditioning
  already-generated options and Undo incur no extra AI request. This writes lyric
  ideas, not a melody or a claim about what will be a hit.

**Writing tools** on a Vocal Studio version carries its latest words, existing
line locks and song direction into a separate Lyric Desk notebook, preserving
an existing Desk draft before import. It does not replace the saved vocal song.
The shared phone review Site still needs this workspace connected; the Mac desk
has responsive layouts, but localhost is not a phone-accessible deployment.

Hook design reference: Berklee's [How to Write Killer Song Hooks](https://online.berklee.edu/takenote/how-to-write-songs-with-killer-hooks/)
(repetition, contrast and placement). Sound reference: the
[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict).

## Future experiments

Keep refining the mumble-to-lyrics workflow first. Later experiments: a private
singing-voice model trained on the artist’s own dry vocals, and a Logic Pro
Audio Unit workflow. These are future directions, not currently installed features.
