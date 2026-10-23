"""Key setup is available only in the explicitly enabled, loopback-only runner."""
import os
import secrets
from pathlib import Path
from threading import Lock

from dotenv import set_key
from flask import Blueprint, abort, current_app, redirect, render_template, request, session, url_for

settings = Blueprint("local_settings", __name__)
_save_lock = Lock()
MODELS = {"gpt-6-luna": "Budget writing", "gpt-5.4-mini": "Alternative writing"}


def enabled():
    return os.getenv("LYRIC_GOAT_LOCAL_SETUP") == "1"


@settings.route("/settings", methods=["GET", "POST"])
def configure():
    if not enabled() or request.remote_addr not in {"127.0.0.1", "::1"}:
        abort(404)
    session.setdefault("setup_csrf", secrets.token_urlsafe(32))
    error = None
    if request.method == "POST":
        token = request.form.get("csrf", "")
        if not secrets.compare_digest(token, session["setup_csrf"]):
            abort(403)
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403)
        key = request.form.get("api_key", "").strip()
        model = request.form.get("model", "gpt-6-luna")
        if model not in MODELS:
            error = "Choose one of the available writing models."
        elif key and (not key.startswith("sk-") or len(key) < 20 or len(key) > 512 or any(c.isspace() for c in key)):
            error = "Enter a valid OpenAI secret key."
        elif not key and not os.getenv("OPENAI_API_KEY"):
            error = "Add your OpenAI key to continue."
        else:
            env_path = Path(current_app.config.get("LOCAL_ENV_FILE", Path(current_app.root_path) / ".env"))
            values = {"LYRIC_GOAT_WRITER": "openai", "OPENAI_GENERATION_MODEL": model}
            if key:
                values["OPENAI_API_KEY"] = key
            try:
                with _save_lock:
                    fd = os.open(env_path, os.O_CREAT | os.O_WRONLY, 0o600)
                    os.close(fd)
                    env_path.chmod(0o600)
                    for name, value in values.items():
                        set_key(env_path, name, value)
                    env_path.chmod(0o600)
                    os.environ.update(values)
            except OSError:
                error = "Could not save settings. Check that the app folder is writable."
            else:
                return redirect(url_for("local_settings.configure", saved="1"), code=303)
    response = current_app.make_response(render_template("settings.html", models=MODELS,
        model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-6-luna"),
        key_present=bool(os.getenv("OPENAI_API_KEY")), csrf=session["setup_csrf"],
        error=error, saved=request.args.get("saved") == "1"))
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response
