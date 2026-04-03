import os
import time
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from services.transcribe import transcribe_audio
from services.analyze import analyze_flow, syllable_rhythm_string
from services.generate import generate_lyrics, generate_single_line
from services.preprocess import preprocess

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

        clean_path = preprocess(audio_path)

        transcription = transcribe_audio(clean_path)
        rough_text = transcription["text"]
        word_timestamps = transcription.get("words", [])

        flow_data = analyze_flow(clean_path, word_timestamps)
        flow_data["rhythm_string"] = syllable_rhythm_string(flow_data.get("flow_map", []))

        versions = generate_lyrics(
            rough_text, flow_data,
            tone=tone, mode=mode, vibe=vibe, gen_mode=gen_mode, key=key
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
