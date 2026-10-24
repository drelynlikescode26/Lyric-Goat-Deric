"""Private, opt-in listening experiments on one immutable audio file.

Results are evidence for the artist to judge, never an automatic accuracy score.
Failures stay failures: a local fallback must not masquerade as an AI result.
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from threading import Lock, Thread
import time
import uuid

import httpx
import numpy as np
from openai import OpenAI
import soundfile as sf
from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, session

from services.analyze import analyze_flow, preanalyze_audio
from services.listen import (apply_listening, listen_acoustic_performance as listen_performance,
                             _detail_prompt, _detail_result)
from services.local_settings import enabled
from services.preprocess import preprocess_branches
from services import cloud_storage

comparisons = Blueprint("comparisons", __name__)
_files = Lock()
_busy = Lock()
_active_runs = set()
MAX_SECONDS = 90
ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = {
    "openai_current": ("OpenAI · current listener", "OpenAI", "Original listener; works inside the detected acoustic phrases."),
    "openai_detail": ("OpenAI · detailed listening", "OpenAI", "Studio default: proposes its own phrases and describes each performed sound."),
    "transcribe": ("GPT Transcribe · words", "OpenAI", "A speech transcription comparison. It does not supply musical phrase timing."),
    "gemini": ("Gemini · detailed listening", "Google", "Uses the same detailed listening task as OpenAI for a provider comparison."),
    "vocalparse": ("VocalParse · experimental", "Local", "Research checkpoint primarily trained on Mandarin singing. English mumble performance is unverified; separate runtime required."),
}


def provider_options():
    options = []
    for key, (label, destination, description) in PROVIDERS.items():
        if key == "gemini":
            ready = bool(os.getenv("GEMINI_API_KEY"))
            model = os.getenv("GEMINI_LISTENING_MODEL", "gemini-3.8-flash")
            reason = "Add a Google Gemini API key in settings."
        elif key == "vocalparse":
            ready = (Path(os.getenv("VOCALPARSE_PYTHON", "/missing/python")).is_file()
                     and Path(os.getenv("VOCALPARSE_CHECKPOINT", "/missing/checkpoint")).is_dir())
            model, reason = "VocalParse-1.7B", "Not enabled: separate research runtime/checkpoint required. This is a Mandarin singing checkpoint, not a validated English listener."
        else:
            ready = bool(os.getenv("OPENAI_API_KEY"))
            model = ("gpt-transcribe" if key == "transcribe"
                     else os.getenv("OPENAI_LISTENING_MODEL", "gpt-audio-1.5"))
            reason = "Add an OpenAI API key in settings."
        options.append(dict(id=key, label=label, destination=destination, description=description,
                            model=model, ready=ready, reason="" if ready else reason))
    return options


def _root():
    return Path(current_app.config.get("COMPARISONS_DIR", cloud_storage.data_root() / "comparisons"))


def _folder(root, identity):
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        abort(404)
    return root / identity


def _load(folder):
    if not (folder / "report.json").exists():
        cloud_storage.ensure_file(folder / "report.json", f"comparisons/{folder.name}/report.json")
    try:
        return json.loads((folder / "report.json").read_text())
    except FileNotFoundError:
        abort(404)


def _save(folder, report):
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as f:
        try:
            os.chmod(f.name, 0o600)
            json.dump(report, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
            os.replace(f.name, folder / "report.json")
            cloud_storage.queue_file(folder / "report.json", f"comparisons/{folder.name}/report.json")
        finally:
            Path(f.name).unlink(missing_ok=True)


def _revision():
    digest = hashlib.sha256()
    for name in ("services/comparisons.py", "services/listen.py", "services/performance.py", "services/analyze.py"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()[:12]


@comparisons.before_request
def private_comparisons():
    if not enabled():
        abort(404)
    if request.remote_addr not in {"127.0.0.1", "::1"}:
        abort(403)
    if request.method == "POST":
        token = request.headers.get("X-Comparison-CSRF", "")
        if not token or not secrets.compare_digest(token, session.get("comparison_csrf", "")):
            abort(403)


@comparisons.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; media-src 'self' blob:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response


@comparisons.route("/compare")
def compare_page():
    session.setdefault("comparison_csrf", secrets.token_urlsafe(32))
    return render_template("compare.html", csrf=session["comparison_csrf"])


@comparisons.route("/api/comparisons/options")
def options_route():
    session.setdefault("comparison_csrf", secrets.token_urlsafe(32))
    return jsonify(providers=provider_options(), max_seconds=MAX_SECONDS, csrf=session["comparison_csrf"])


@comparisons.route("/api/comparisons")
def history():
    rows = []
    for path in _root().glob("*/report.json"):
        try:
            report = json.loads(path.read_text())
            rows.append({k: report[k] for k in ("id", "title", "created_at", "duration")})
        except (OSError, ValueError, KeyError):
            continue
    return jsonify(comparisons=sorted(rows, key=lambda x: x["created_at"], reverse=True))


@comparisons.route("/api/comparisons", methods=["POST"])
def upload():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify(error="Choose an audio take first."), 400
    try:
        bpm = float(request.form.get("bpm", "0") or 0)
        if not math.isfinite(bpm) or bpm < 0 or bpm > 300:
            raise ValueError
    except ValueError:
        return jsonify(error="BPM must be between 0 and 300. Use 0 for an estimate."), 400
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = uuid.uuid4().hex
    folder = _folder(root, identity)
    folder.mkdir(mode=0o700)
    source = folder / "source.upload"
    audio.save(source)
    source.chmod(0o600)
    branches = {}
    try:
        branches = preprocess_branches(str(source))
        take = Path(branches["melody_path"])
        info = sf.info(take)
        if not 0 < info.duration <= MAX_SECONDS + .01:
            return jsonify(error=f"Use a vocal section of {MAX_SECONDS} seconds or less."), 422
        samples, _ = sf.read(take)
        if not samples.size or float(np.max(np.abs(samples))) < 1e-5:
            return jsonify(error="No vocal detected. Choose a clearer take."), 422
        pre = preanalyze_audio(str(take), manual_bpm=bpm)
        flow = analyze_flow(str(take), [], pre_data=pre, hum_mode=True)
        if not flow.get("phrase_map"):
            return jsonify(error="No vocal phrases detected. Choose a clearer take."), 422
        destination = folder / "take.wav"
        take.replace(destination)
        destination.chmod(0o600)
        cloud_storage.queue_file(destination, f"comparisons/{identity}/take.wav")
        report = dict(id=identity, title=(request.form.get("title") or audio.filename)[:120],
                      created_at=time.time(), duration=round(info.duration, 3),
                      audio_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      reference_revision=_revision(), bpm=bpm, reference=flow["phrase_map"], runs=[])
        _save(folder, report)
        return jsonify(report), 201
    except Exception:
        return jsonify(error="Could not read this audio. Try a WAV, M4A, MP3, or WebM recording."), 422
    finally:
        for path in [source, *[Path(v) for v in branches.values() if isinstance(v, str)]]:
            path.unlink(missing_ok=True)
        if not (folder / "report.json").exists():
            (folder / "take.wav").unlink(missing_ok=True)
            folder.rmdir()


@comparisons.route("/api/comparisons/<identity>")
def report_route(identity):
    with _files:
        folder = _folder(_root(), identity)
        report = _load(folder)
        changed = False
        for run in report["runs"]:
            if run["status"] == "running" and run["id"] not in _active_runs:
                run["status"] = "failed"
                run["error"] = "The app restarted during this run. Your saved take can be compared again."
                for row in run["results"]:
                    if row["status"] == "pending":
                        row.update(status="failed", error="Run interrupted by restart.", phrases=[])
                changed = True
        if changed:
            _save(folder, report)
        return jsonify(report)


@comparisons.route("/api/comparisons/<identity>/audio")
def audio_route(identity):
    folder = _folder(_root(), identity)
    _load(folder)
    path = cloud_storage.ensure_file(folder / "take.wav", f"comparisons/{identity}/take.wav")
    if path is None: abort(404)
    return cloud_storage.audio_response(path, mimetype="audio/wav", key=f"comparisons/{identity}/take.wav")


@comparisons.route("/api/comparisons/<identity>/runs", methods=["POST"])
def start_run(identity):
    folder = _folder(_root(), identity)
    report = _load(folder)
    selected = (request.get_json(silent=True) or {}).get("providers")
    if (not isinstance(selected, list) or not selected or len(selected) > len(PROVIDERS)
            or any(not isinstance(x, str) or x not in PROVIDERS for x in selected)
            or len(set(selected)) != len(selected)):
        return jsonify(error="Choose one or more distinct listeners."), 400
    available = {p["id"]: p for p in provider_options()}
    if any(not available[p]["ready"] for p in selected):
        return jsonify(error="A selected listener needs setup. Check its settings first."), 400
    if not _busy.acquire(blocking=False):
        return jsonify(error="A comparison is already running. Wait for it to finish."), 409
    try:
        run = dict(id=uuid.uuid4().hex, created_at=time.time(), revision=_revision(),
                   audio_sha256=report["audio_sha256"], status="running",
                   results=[dict(**available[p], status="pending") for p in selected], review={})
        with _files:
            report = _load(folder)
            report["runs"].append(run)
            _save(folder, report)
            _active_runs.add(run["id"])
        Thread(target=_run, args=(folder, report, run), daemon=True).start()
        return jsonify(run), 202
    except Exception:
        _active_runs.discard(run["id"])
        _busy.release()
        raise


def _run(folder, report, run):
    with cloud_storage.pinned(folder / "take.wav"):
        _run_pinned(folder, report, run)


def _run_pinned(folder, report, run):
    try:
        cloud_storage.ensure_file(folder / "take.wav", f"comparisons/{folder.name}/take.wav")
        if report["reference_revision"] != run["revision"]:
            # Reuse the exact audio, but let detector tweaks participate in a
            # new experiment. Previous runs and their timings stay untouched.
            report = deepcopy(report)
            audio = str(folder / "take.wav")
            pre = preanalyze_audio(audio, manual_bpm=report["bpm"])
            report["reference"] = analyze_flow(audio, [], pre_data=pre, hum_mode=True)["phrase_map"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = {pool.submit(_one, row, folder / "take.wav", report): row["id"] for row in run["results"]}
            for future in as_completed(tasks):
                result = future.result()
                with _files:
                    saved = _load(folder)
                    current = next(r for r in saved["runs"] if r["id"] == run["id"])
                    current["results"] = [result if p["id"] == result["id"] else p for p in current["results"]]
                    if all(p["status"] != "pending" for p in current["results"]):
                        current["status"] = "complete"
                    _save(folder, saved)
    except Exception:
        with _files:
            saved = _load(folder)
            current = next(r for r in saved["runs"] if r["id"] == run["id"])
            current["status"] = "failed"
            current["error"] = "The comparison stopped before finishing. Your saved take is available for another run."
            for row in current["results"]:
                if row["status"] == "pending":
                    row.update(status="failed", error="Comparison interrupted.", phrases=[])
            _save(folder, saved)
    finally:
        _active_runs.discard(run["id"])
        _busy.release()


def _one(option, audio, report):
    started = time.monotonic()
    result = dict(option, audio_sha256=report["audio_sha256"], status="ok")
    try:
        key = option["id"]
        if key == "openai_current":
            listening = listen_performance(str(audio), report["reference"])
            if listening.get("status") != "ok":
                raise ValueError("Listener unavailable")
            if len(listening.get("phrases", [])) != len(report["reference"]):
                raise ValueError("Incomplete listening response")
            flow = apply_listening({"phrase_map": deepcopy(report["reference"])}, listening)
            result.update(phrases=flow["phrase_map"], feel=listening.get("feel", []),
                          timing_note="Uses the shared acoustic phrase windows; AI describes sounds inside them.")
        elif key == "transcribe":
            with audio.open("rb") as f:
                response = OpenAI(timeout=60, max_retries=0).audio.transcriptions.create(
                    model=option["model"], file=f,
                    prompt="A rough sung topline. Transcribe only clearly audible words; do not invent lyrics for humming or gibberish.")
            result.update(transcript=str(response.text)[:16000], phrases=[],
                          timing_note="Speech transcript only. No musical phrase boundaries or syllable timing supplied.")
        elif key == "vocalparse":
            if report["duration"] > 30:
                return dict(result, status="failed", error="VocalParse's published checkpoint needs sections of 30 seconds or less. Other listeners keep the full saved take.",
                            phrases=[], elapsed_seconds=round(time.monotonic() - started, 2))
            response = subprocess.run([os.environ["VOCALPARSE_PYTHON"], str(ROOT / "scripts/vocalparse_compare.py"),
                "--audio", str(audio), "--checkpoint", os.environ["VOCALPARSE_CHECKPOINT"]],
                capture_output=True, text=True, timeout=180, check=True)
            result.update(json.loads(response.stdout))
        else:
            prompt = _detail_prompt(report["duration"])
            encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
            if key == "gemini":
                model = option["model"]
                if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
                    raise ValueError("Invalid model")
                response = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60,
                    json={"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "audio/wav", "data": encoded}}]}],
                          "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 6000}})
                response.raise_for_status()
                payload = response.json()
                candidate = payload["candidates"][0]
                if candidate.get("finishReason") != "STOP":
                    raise ValueError("Incomplete response")
                raw = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
            else:
                response = OpenAI(timeout=60, max_retries=0).chat.completions.create(
                    model=option["model"], modalities=["text"], max_completion_tokens=6000,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"format": "wav", "data": encoded}}]}])
                if response.choices[0].finish_reason != "stop":
                    raise ValueError("Incomplete response")
                raw = response.choices[0].message.content or ""
            result.update(_detail_result(raw, report["duration"]))
    except Exception as exc:
        # Never expose provider error bodies, prompts, API keys, or local paths.
        result.update(status="failed", error="This listener did not return a complete usable result. Check model access/settings or try a shorter take.",
                      error_type=type(exc).__name__, phrases=[])
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return result




@comparisons.route("/api/comparisons/<identity>/boundaries", methods=["POST"])
def save_boundaries(identity):
    data = request.get_json(silent=True)
    if not isinstance(data, dict): return jsonify(error="Send the phrase-break times first."), 400
    cuts = data.get("cuts")
    if (not isinstance(cuts, list) or len(cuts) > 64 or any(type(c) not in {int, float} or not math.isfinite(c) for c in cuts)):
        return jsonify(error="Use up to 64 phrase breaks, in seconds."), 400
    folder = _folder(_root(), identity)
    with _files:
        report = _load(folder)
        if data.get("audio_sha256") != report["audio_sha256"]:
            return jsonify(error="This recording changed. Reopen the correct take before marking breaks."), 409
        previous = report.get("boundary_reference", {})
        if type(data.get("revision")) is not int or data["revision"] != previous.get("revision", 0):
            return jsonify(error="Phrase marks were updated in another tab. Reopen this take before editing them."), 409
        if any(not .03 < c < report["duration"] - .03 for c in cuts) or any(b-a < .1 for a,b in zip(cuts,cuts[1:])):
            return jsonify(error="Put breaks inside the recording, in increasing order, at least 0.10 seconds apart."), 400
        report["boundary_reference"] = dict(cuts=[round(c, 3) for c in cuts], revision=previous.get("revision", 0)+1,
            updated_at=time.time(), audio_sha256=report["audio_sha256"], source="artist_marks")
        _save(folder, report)
    return jsonify(report)


@comparisons.route("/api/comparisons/<identity>/review", methods=["POST"])
def review(identity):
    data = request.get_json(silent=True) or {}
    folder = _folder(_root(), identity)
    with _files:
        report = _load(folder)
        run = next((r for r in report["runs"] if r["id"] == data.get("run_id")), None)
        if not run or run["status"] != "complete":
            return jsonify(error="Wait for this run to finish before saving a review."), 400
        candidates = {p["id"] for p in run["results"] if p["status"] == "ok"}
        provider = data.get("provider")
        if provider not in candidates:
            return jsonify(error="Choose a successful result to review."), 400
        scores = data.get("scores", {})
        if (not isinstance(scores, dict) or any(k not in scores or (scores[k] is not None
                and (type(scores[k]) is not int or not 1 <= scores[k] <= 5))
                for k in ("flow", "sounds", "syllables"))):
            return jsonify(error="Rate each item from 1 to 5, or choose not applicable."), 400
        run["review"].setdefault("ratings", {})[provider] = dict(
            scores={k: scores[k] for k in ("flow", "sounds", "syllables")}, notes=str(data.get("notes", ""))[:2000])
        if data.get("preferred") is True:
            run["review"]["preferred"] = provider
        _save(folder, report)
    return jsonify(report)
