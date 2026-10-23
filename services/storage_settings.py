"""Loopback-only cloud setup; credentials never leave the Mac settings file."""
import os
from pathlib import Path
import secrets
import re
from threading import Lock

from dotenv import set_key
from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, session, url_for
from services import cloud_storage as storage
from services.local_settings import enabled

storage_settings = Blueprint("storage_settings", __name__)
_lock = Lock()


@storage_settings.before_request
def private():
    if not enabled() or request.remote_addr not in {"127.0.0.1", "::1"}: abort(404)
    if request.method == "POST":
        token = request.form.get("csrf", "")
        if not session.get("storage_csrf") or not secrets.compare_digest(token, session["storage_csrf"]): abort(403)
        if request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"): abort(403)


@storage_settings.route("/api/storage/status")
def status():
    try: data = storage.status()
    except Exception: data = {"enabled": False, "state": "error", "pending": 0, "error": "Cloud settings need attention. Local saves are kept."}
    response = jsonify(data); response.headers["Cache-Control"] = "no-store"
    return response


@storage_settings.route("/storage", methods=["GET", "POST"])
def configure():
    session.setdefault("storage_csrf", secrets.token_urlsafe(32))
    error = None; message = request.args.get("message", "")
    if request.method == "POST":
        action = request.form.get("action", "connect")
        try:
            if action == "connect":
                values = {name: request.form.get(name, "").strip() or os.getenv(name, "") for name in
                    ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")}
                if (not re.fullmatch(r"[A-Za-z0-9]{16,128}", values["R2_ACCESS_KEY_ID"])
                        or not re.fullmatch(r"[A-Za-z0-9/+_=.-]{20,256}", values["R2_SECRET_ACCESS_KEY"])):
                    raise storage.StorageError("Enter the R2 Access Key ID and Secret Access Key from your bucket-scoped token.")
                storage.probe(dict(values))
                values["LYRIC_GOAT_STORAGE"] = "r2"
                env_path = Path(current_app.config.get("LOCAL_ENV_FILE", Path(current_app.root_path) / ".env"))
                with _lock:
                    fd = os.open(env_path, os.O_CREAT | os.O_WRONLY, 0o600); os.close(fd); env_path.chmod(0o600)
                    for name, value in values.items(): set_key(env_path, name, value)
                    env_path.chmod(0o600); os.environ.update(values)
                count = storage.migrate_saved()
                message = f"Connection verified. {count} saved library files queued for cloud storage."
            elif action == "retry":
                count = storage.migrate_saved()
                message = f"Checking {count} saved library files. Pending cloud saves will retry."
            elif action == "refresh":
                instance = storage.library()
                if not instance: raise storage.StorageError("Connect R2 first.")
                count = instance.restore_catalog()
                message = f"Opened {count} missing song or comparison records from the cloud. Existing local edits were kept."
            elif action == "clear-cache":
                instance = storage.library()
                if not instance: raise storage.StorageError("Connect R2 first.")
                removed = instance.prune(clear=True)
                message = f"Freed {removed / (1024*1024):.1f} MB of verified audio cache. Cloud copies and pending saves were kept."
            else: abort(400)
        except storage.StorageError as exc: error = str(exc)
        except Exception: error = "Storage setup could not finish. Existing files and settings are kept; check your connection and available disk space."
        if not error: return redirect(url_for("storage_settings.configure", message=message), code=303)
    try: state = storage.status()
    except Exception: state = {"enabled": False, "error": "Cloud settings need attention.", "pending": 0, "cache_bytes": 0}
    response = current_app.make_response(render_template("storage.html", state=state, csrf=session["storage_csrf"],
        error=error, message=message, account=os.getenv("R2_ACCOUNT_ID", ""), bucket=os.getenv("R2_BUCKET", "")))
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response
