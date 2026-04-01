import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from services.transcribe import transcribe_audio
from services.analyze import analyze_flow, syllable_rhythm_string
from services.generate import generate_lyrics

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Main endpoint. Accepts:
      - audio: file upload OR base64 blob from mic recording
      - tone: melodic | aggressive | simple | punchlines
      - mode: hook | verse | story
      - vibe: sad | hype | introspective | love
    """
    # --- Handle audio input ---
    audio_path = None
    tmp_file = None

    if "audio" in request.files:
        f = request.files["audio"]
        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400

        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "webm"
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=f".{ext}", dir=UPLOAD_FOLDER, delete=False
        )
        f.save(tmp_file.name)
        audio_path = tmp_file.name

    else:
        return jsonify({"error": "No audio provided"}), 400

    try:
        tone = request.form.get("tone", "melodic")
        mode = request.form.get("mode", "verse")
        vibe = request.form.get("vibe", "introspective")

        # Step 1: Transcribe
        transcription = transcribe_audio(audio_path)
        rough_text = transcription["text"]
        word_timestamps = transcription.get("words", [])

        # Step 2: Analyze flow
        flow_data = analyze_flow(audio_path, word_timestamps)
        flow_data["rhythm_string"] = syllable_rhythm_string(flow_data.get("flow_map", []))

        # Step 3: Generate lyrics
        versions = generate_lyrics(rough_text, flow_data, tone=tone, mode=mode, vibe=vibe)

        return jsonify({
            "success": True,
            "rough_text": rough_text,
            "flow": {
                "tempo_bpm": flow_data["tempo_bpm"],
                "flow_style": flow_data["flow_style"],
                "syllable_count": flow_data["syllable_count"],
            },
            "versions": versions,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
