import os
import time
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv

load_dotenv()

from services.transcribe import transcribe_audio
from services.analyze import analyze_flow, preanalyze_audio, syllable_rhythm_string
from services.generate import generate_lyrics, generate_single_line
from services.preprocess import preprocess
from services import projects

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

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
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "No audio provided"}), 400

    f = request.files["audio"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "webm"
    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", dir=UPLOAD_FOLDER, delete=False)
    f.save(tmp_file.name)
    audio_path = tmp_file.name
    clean_path = None

    try:
        tone     = request.form.get("tone", "melodic")
        mode     = request.form.get("mode", "verse")
        vibe     = request.form.get("vibe", "introspective")
        gen_mode = request.form.get("gen_mode", "cadence")
        key      = request.form.get("key", "auto")
        genre      = request.form.get("genre", "hiphop")
        manual_bpm = float(request.form.get("manual_bpm", "0") or "0")
        hum_mode   = request.form.get("hum_mode", "false").lower() == "true"

        clean_path = preprocess(audio_path)

        if hum_mode:
            # Skip transcription entirely — pure audio analysis only.
            # Saves ~3-5s and avoids confusing the model with non-speech audio.
            pre_data        = preanalyze_audio(clean_path, manual_bpm=manual_bpm)
            rough_text      = ""
            word_timestamps = []
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                t_future = ex.submit(transcribe_audio, clean_path)
                a_future = ex.submit(preanalyze_audio, clean_path, manual_bpm)
            transcription   = t_future.result()
            pre_data        = a_future.result()
            rough_text      = transcription["text"]
            word_timestamps = transcription.get("words", [])

        flow_data = analyze_flow(clean_path, word_timestamps, pre_data=pre_data, hum_mode=hum_mode)
        flow_data["rhythm_string"] = syllable_rhythm_string(flow_data.get("flow_map", []))

        versions = generate_lyrics(
            rough_text, flow_data,
            tone=tone, mode=mode, vibe=vibe, gen_mode=gen_mode, key=key, genre=genre
        )

        return jsonify({
            "success": True,
            "rough_text": rough_text,
            "melody_mode": flow_data.get("melody_mode", False),
            "phrase_map": flow_data.get("phrase_map", []),
            "detected_key": flow_data.get("detected_key"),
            "vowel_family": flow_data.get("vowel_family"),
            "is_repetitive": flow_data.get("is_repetitive", False),
            "debug_phrases": flow_data.get("debug_phrases", []),
            "flow": {
                "tempo_bpm": flow_data["tempo_bpm"],
                "flow_style": flow_data["flow_style"],
                "syllable_count": flow_data["syllable_count"],
                "duration": flow_data.get("duration"),
            },
            "versions": versions,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for path in (audio_path, clean_path):
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

        new_line = generate_single_line(
            bar_index, syllable_count, context_lines, locked_lines,
            rough_text, flow_data, tone, mode, vibe, gen_mode, key
        )

        return jsonify({"success": True, "line": new_line})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Song Workspace API ───────────────────────────────────────────────────────

@app.route("/api/songs", methods=["GET"])
def api_list_songs():
    return jsonify({"success": True, "songs": projects.list_songs()})


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
    app.run(debug=True, host="0.0.0.0", port=5000)
