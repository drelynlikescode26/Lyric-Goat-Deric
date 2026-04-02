import os
import time
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from services.transcribe import transcribe_audio
from services.analyze import analyze_flow, syllable_rhythm_string
from services.generate import generate_lyrics
from services.preprocess import preprocess

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Clean up orphaned temp files older than 1 hour at startup
_now = time.time()
for _f in UPLOAD_FOLDER.iterdir():
    if _f.is_file() and (_now - _f.stat().st_mtime) > 3600:
        _f.unlink(missing_ok=True)


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

    clean_path = None
    try:
        tone = request.form.get("tone", "melodic")
        mode = request.form.get("mode", "verse")
        vibe = request.form.get("vibe", "introspective")

        # Step 1: Preprocess (normalize, resample to 16kHz mono WAV)
        clean_path = preprocess(audio_path)

        # Step 2: Transcribe
        transcription = transcribe_audio(clean_path)
        rough_text = transcription["text"]
        word_timestamps = transcription.get("words", [])

        # Step 3: Analyze flow
        flow_data = analyze_flow(clean_path, word_timestamps)
        flow_data["rhythm_string"] = syllable_rhythm_string(flow_data.get("flow_map", []))

        # Step 4: Generate lyrics
        versions = generate_lyrics(rough_text, flow_data, tone=tone, mode=mode, vibe=vibe)

        return jsonify({
            "success": True,
            "rough_text": rough_text,
            "melody_mode": flow_data.get("melody_mode", False),
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
        for path in (audio_path, clean_path):
            if path and os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
