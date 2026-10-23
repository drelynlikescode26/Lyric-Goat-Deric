import os
import time
import tempfile
import secrets
import soundfile as sf
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv

load_dotenv()

from services.transcribe import TranscriptionError, transcribe_audio
from services.analyze import analyze_flow, preanalyze_audio, syllable_rhythm_string
from services.generate import GenerationError, generate_lyrics, generate_single_line
from services.listen import listen_performance, apply_listening
from services.preprocess import preprocess_branches
from services import projects
from services import writing_profile
from services.local_settings import settings, enabled as local_setup_enabled

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
if local_setup_enabled():
    app.config["TRUSTED_HOSTS"] = ["127.0.0.1", "localhost", "[::1]"]
app.register_blueprint(settings)


@app.before_request
def protect_local_app():
    if local_setup_enabled():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify({"error": "Local access only."}), 403
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if ((origin and origin != request.host_url.rstrip("/"))
                    or request.headers.get("Sec-Fetch-Site") == "cross-site"):
                return jsonify({"error": "Open Lyric Goat directly to continue."}), 403

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_now = time.time()
for _f in UPLOAD_FOLDER.iterdir():
    if _f.is_file() and (_now - _f.stat().st_mtime) > 3600:
        _f.unlink(missing_ok=True)


@app.route("/")
def index():
    return render_template("index.html", local_setup=local_setup_enabled())


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "lyric-goat"})


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "No audio provided"}), 400

    f = request.files["audio"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "webm"
    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", dir=UPLOAD_FOLDER, delete=False)
    f.save(tmp_file.name)
    audio_path = tmp_file.name
    derived_paths = []

    try:
        tone     = request.form.get("tone", "melodic")
        mode     = request.form.get("mode", "verse")
        vibe     = request.form.get("vibe", "introspective")
        gen_mode = request.form.get("gen_mode", "cadence")
        key      = request.form.get("key", "auto")
        genre      = request.form.get("genre", "hiphop")
        manual_bpm = float(request.form.get("manual_bpm", "0") or "0")
        hum_mode   = request.form.get("hum_mode", "false").lower() == "true"

        branches = preprocess_branches(audio_path)
        speech_path = branches["speech_path"]
        melody_path = branches["melody_path"]
        derived_paths.extend([speech_path, melody_path])
        if sf.info(melody_path).duration > 90:
            return jsonify({"error": "Use a vocal section of 90 seconds or less."}), 422
        samples, _ = sf.read(melody_path)
        if not samples.size or np.max(np.abs(samples)) < 1e-5:
            return jsonify({"error": "No vocal detected. Try a closer, clearer vocal take."}), 422
        del samples
        transcription_diagnostics = {
            "mode": "skipped",
            "status": "skipped",
            "reason": "hum_mode",
            "errors": [],
        }

        if hum_mode:
            # Skip transcription entirely — pure audio analysis only.
            # Saves ~3-5s and avoids confusing the model with non-speech audio.
            pre_data        = preanalyze_audio(melody_path, manual_bpm=manual_bpm)
            rough_text      = ""
            word_timestamps = []
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                t_future = ex.submit(transcribe_audio, speech_path)
                a_future = ex.submit(preanalyze_audio, melody_path, manual_bpm)
            pre_data        = a_future.result()
            try:
                transcription = t_future.result()
                rough_text = transcription["text"]
                word_timestamps = transcription.get("words", [])
                transcription_diagnostics = transcription.get("diagnostics", {})
            except TranscriptionError:
                # Mumbles and hums do not need recognizable speech to be usable.
                rough_text, word_timestamps = "", []
                transcription_diagnostics = {"status": "unavailable", "mode": "audio_only"}

        flow_data = analyze_flow(melody_path, word_timestamps, pre_data=pre_data, hum_mode=hum_mode)
        flow_data["rhythm_string"] = syllable_rhythm_string(flow_data.get("flow_map", []))
        if not flow_data.get("phrase_map"):
            return jsonify({"error": "No vocal detected. Try a closer, clearer vocal take."}), 422
        listening = listen_performance(melody_path, flow_data["phrase_map"])
        apply_listening(flow_data, listening)
        if not rough_text:
            rough_text = " ".join(p.get("heard_words", "") for p in flow_data["phrase_map"]).strip()

        versions = generate_lyrics(
            rough_text, flow_data,
            tone=tone, mode=mode, vibe=vibe, gen_mode=gen_mode, key=key, genre=genre
        )

        return jsonify({
            "success": True,
            "rough_text": rough_text,
            "transcription": transcription_diagnostics,
            "listening": listening,
            "melody_mode": flow_data.get("melody_mode", False),
            "phrase_map": flow_data.get("phrase_map", []),
            "detected_key": flow_data.get("detected_key"),
            "vowel_family": flow_data.get("vowel_family"),
            "is_repetitive": flow_data.get("is_repetitive", False),
            "debug_phrases": flow_data.get("debug_phrases", []),
            "note_provider": flow_data.get("note_provider", "pyin"),
            "flow": {
                "tempo_bpm": flow_data["tempo_bpm"],
                "flow_style": flow_data["flow_style"],
                "syllable_count": flow_data["syllable_count"],
                "duration": flow_data.get("duration"),
            },
            "versions": versions,
        })

    except GenerationError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for path in [audio_path, *derived_paths]:
            if path and os.path.exists(path):
                os.unlink(path)


@app.route("/regenerate-line", methods=["POST"])
def regenerate_line():
    """
    Regenerate a single lyric bar.
    Body JSON:
      bar_index, syllable_count, context_lines[], locked_lines{},
      rough_text, flow_data{}, tone, mode, vibe, gen_mode, key
    """
    try:
        data = request.get_json()
        bar_index      = int(data["bar_index"])
        syllable_count = int(data["syllable_count"])
        context_lines  = data.get("context_lines", [])
        locked_lines   = data.get("locked_lines", {})
        rough_text     = data.get("rough_text", "")
        flow_data      = data.get("flow_data", {})
        tone           = data.get("tone", "melodic")
        mode           = data.get("mode", "verse")
        vibe           = data.get("vibe", "introspective")
        gen_mode       = data.get("gen_mode", "cadence")
        key            = data.get("key", "auto")
        genre          = data.get("genre", "hiphop")

        new_line = generate_single_line(
            bar_index, syllable_count, context_lines, locked_lines,
            rough_text, flow_data, tone, mode, vibe, gen_mode, key, genre
        )

        return jsonify({"success": True, "line": new_line})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/regenerate-all", methods=["POST"])
def regenerate_all():
    """Generate new versions from an already-reviewed phrase map.

    This avoids uploading and transcribing the audio again when the artist only
    changed syllable slots or wants fresh lyric candidates.
    """
    try:
        data = request.get_json(silent=True) or {}
        flow_data = data.get("flow_data") or {}
        phrase_map = flow_data.get("phrase_map") or []
        if not phrase_map:
            return jsonify({"error": "No phrase slots provided"}), 400

        versions = generate_lyrics(
            data.get("rough_text", ""),
            flow_data,
            tone=data.get("tone", "melodic"),
            mode=data.get("mode", "verse"),
            vibe=data.get("vibe", "introspective"),
            gen_mode=data.get("gen_mode", "cadence"),
            key=data.get("key", "auto"),
            genre=data.get("genre", "hiphop"),
            locked_versions=data.get("locked_versions", {}),
        )
        return jsonify({"success": True, "versions": versions, "phrase_map": phrase_map})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Song Workspace API ───────────────────────────────────────────────────────

@app.route("/api/songs", methods=["GET"])
def api_list_songs():
    return jsonify({"success": True, "songs": projects.list_songs()})


@app.route("/api/writing-feedback", methods=["GET", "POST"])
def api_writing_feedback():
    if request.method == "GET":
        limit = request.args.get("limit", 50, type=int)
        return jsonify({"success": True, "feedback": writing_profile.recent_feedback(limit)})
    try:
        entry = writing_profile.record_feedback(request.get_json(silent=True) or {})
        return jsonify({"success": True, "feedback": entry}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/songs", methods=["POST"])
def api_create_song():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "Untitled")
    song = projects.create_song(
        title,
        bpm=int(data.get("bpm", 90) or 90),
        key=data.get("key", "auto"),
        genre=data.get("genre", "hiphop"),
    )
    return jsonify({"success": True, "song": song})


@app.route("/api/songs/<song_id>", methods=["GET"])
def api_get_song(song_id):
    song = projects.get_song(song_id)
    if song is None:
        return jsonify({"error": "Song not found"}), 404
    return jsonify({"success": True, "song": song})


@app.route("/api/songs/<song_id>", methods=["PUT"])
def api_update_song(song_id):
    data = request.get_json(silent=True) or {}
    song = projects.update_song(
        song_id,
        title=data.get("title"),
        bpm=data.get("bpm"),
        key=data.get("key"),
        genre=data.get("genre"),
    )
    if song is None:
        return jsonify({"error": "Song not found"}), 404
    return jsonify({"success": True, "song": song})


@app.route("/api/songs/<song_id>", methods=["DELETE"])
def api_delete_song(song_id):
    ok = projects.delete_song(song_id)
    return jsonify({"success": ok})


@app.route("/api/songs/<song_id>/sections", methods=["POST"])
def api_add_section(song_id):
    data = request.get_json(silent=True) or {}
    section = projects.add_section(song_id, data)
    if section is None:
        return jsonify({"error": "Song not found"}), 404
    return jsonify({"success": True, "section": section})


@app.route("/api/songs/<song_id>/sections/<section_id>", methods=["PUT"])
def api_update_section(song_id, section_id):
    data = request.get_json(silent=True) or {}
    section = projects.update_section(song_id, section_id, data)
    if section is None:
        return jsonify({"error": "Section not found"}), 404
    return jsonify({"success": True, "section": section})


@app.route("/api/songs/<song_id>/sections/<section_id>", methods=["DELETE"])
def api_delete_section(song_id, section_id):
    ok = projects.delete_section(song_id, section_id)
    return jsonify({"success": ok})


@app.route("/api/songs/<song_id>/reorder", methods=["POST"])
def api_reorder_sections(song_id):
    data = request.get_json(silent=True) or {}
    song = projects.reorder_sections(song_id, data.get("order", []))
    if song is None:
        return jsonify({"error": "Song not found"}), 404
    return jsonify({"success": True, "song": song})


@app.route("/api/songs/<song_id>/sections/<section_id>/audio", methods=["POST"])
def api_save_section_audio(song_id, section_id):
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "No audio provided"}), 400
    f = request.files["audio"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "webm"
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", dir=UPLOAD_FOLDER, delete=False)
    f.save(tmp.name)
    try:
        filename = projects.save_section_audio(song_id, section_id, tmp.name, ext)
        if filename is None:
            return jsonify({"error": "Song or section not found"}), 404
        return jsonify({"success": True, "audio_file": filename})
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.route("/api/songs/<song_id>/sections/<section_id>/audio", methods=["GET"])
def api_get_section_audio(song_id, section_id):
    song = projects.get_song(song_id)
    if song is None:
        return jsonify({"error": "Song not found"}), 404
    section = next((s for s in song.get("sections", []) if s["id"] == section_id), None)
    if section is None or not section.get("audio_file"):
        return jsonify({"error": "No audio"}), 404
    path = projects.audio_path(section["audio_file"])
    if path is None:
        return jsonify({"error": "Audio file missing"}), 404
    return send_file(str(path))


@app.route("/api/songs/<song_id>/export", methods=["GET"])
def api_export_song(song_id):
    text = projects.assemble_song_text(song_id)
    if not text:
        return jsonify({"error": "Song not found"}), 404
    return jsonify({"success": True, "text": text})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8765)
