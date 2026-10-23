"""Private R2 library with a durable outbox and a bounded, verified audio cache.

Only saved library files enter this service. API keys, uploads, and app files never do.
Local writes finish first; a single background worker sends immutable snapshots.
"""
from contextlib import contextmanager
import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
from threading import Event, RLock, Thread
import time
import uuid

PREFIX = "lyric-goat/v1/"
MAX_FILE = 60 * 1024 * 1024
_instances = {}
_instances_lock = RLock()


class StorageError(Exception):
    pass


def data_root():
    return Path(os.getenv("LYRIC_GOAT_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).resolve()


def configured():
    return os.getenv("LYRIC_GOAT_STORAGE") == "r2" and all(os.getenv(k) for k in
        ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def valid_key(key):
    parts = PurePosixPath(key).parts
    if (not parts or key.startswith("/") or any(p in {"..", "."} for p in parts)
            or "\\" in key or key != "/".join(parts)):
        raise StorageError("Invalid library path.")
    if not ((len(parts) == 2 and parts[0] == "songs" and re.fullmatch(r"[a-f0-9]{12}\.json", parts[1]))
            or (len(parts) == 2 and parts[0] == "notebooks" and re.fullmatch(r"[a-f0-9]{32}\.json", parts[1]))
            or (len(parts) == 2 and parts[0] == "audio" and re.fullmatch(r"[a-f0-9_]+\.(wav|mp3|m4a|webm|ogg|mp4|flac)", parts[1]))
            or (len(parts) == 3 and parts[0] == "comparisons" and re.fullmatch(r"[a-f0-9]{32}", parts[1]) and parts[2] in {"take.wav", "report.json"})
            or key == "profile/writing_feedback.jsonl"):
        raise StorageError("This file is outside the saved library.")
    return key


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            os.chmod(f.name, 0o600); f.write(content); f.flush(); os.fsync(f.fileno()); f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def make_client(values=None):
    values = values or os.environ
    account = values.get("R2_ACCOUNT_ID", "")
    bucket = values.get("R2_BUCKET", "")
    if not re.fullmatch(r"[a-f0-9]{32}", account) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise StorageError("Check your Cloudflare account ID and bucket name.")
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=values["R2_ACCESS_KEY_ID"], aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4", connect_timeout=8,
            read_timeout=45, retries={"max_attempts": 1}, s3={"addressing_style": "path"},
            request_checksum_calculation="when_required", response_checksum_validation="when_required"))


def probe(values):
    """Test a disposable object; never probe with a user's recording."""
    try:
        client = make_client(values); bucket = values["R2_BUCKET"]
        key = PREFIX + "connection-tests/" + uuid.uuid4().hex
        content = b"Lyric Goat private storage connection test."
        client.put_object(Bucket=bucket, Key=key, Body=content, ContentType="text/plain")
        response = client.get_object(Bucket=bucket, Key=key)
        try:
            if response["Body"].read() != content:
                raise StorageError("The storage connection could not be verified.")
        finally:
            response["Body"].close()
        client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        # Provider exceptions may contain credentials or private object names.
        raise StorageError("Could not verify R2. Check the account, bucket, object read/write permission, and connection.") from None


class Library:
    def __init__(self, root, client, bucket, namespace="test", cache_bytes=50*1024*1024):
        self.root = Path(root).resolve(); self.client = client; self.bucket = bucket
        self.cache_bytes = cache_bytes; self.lock = RLock(); self.pins = {}; self.wake = Event()
        self.worker = None; self.error = ""; self.closed = False
        self.state_dir = self.root / ".cloud" / namespace
        self.pending = self.state_dir / "pending"; self.pending.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.state_dir / "state.sqlite", check_same_thread=False)
        os.chmod(self.state_dir / "state.sqlite", 0o600)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS objects(key TEXT PRIMARY KEY,path TEXT,sha TEXT,verified TEXT,touched REAL);
            CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY,sha TEXT,snapshot TEXT,operation TEXT,created REAL);
            CREATE TABLE IF NOT EXISTS deleted(key TEXT PRIMARY KEY);
        """)
        self.db.commit()

    def enqueue(self, path, key):
        valid_key(key); path = Path(path).resolve()
        content = path.read_bytes()
        if len(content) > MAX_FILE:
            raise StorageError("This saved file is too large for the cloud connection.")
        sha = hashlib.sha256(content).hexdigest()
        with self.lock:
            old = self.db.execute("SELECT sha,verified FROM objects WHERE key=?", (key,)).fetchone()
            if old == (sha, sha) and not self.db.execute("SELECT 1 FROM deleted WHERE key=?", (key,)).fetchone():
                return
            snapshot = self.pending / (sha + ".blob")
            if not snapshot.exists(): atomic_bytes(snapshot, content)
            self.db.execute("INSERT INTO objects VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET path=excluded.path,sha=excluded.sha,touched=excluded.touched",
                (key, str(path), sha, "", time.time()))
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?)", (key, sha, str(snapshot), "put", time.time()))
            self.db.execute("DELETE FROM deleted WHERE key=?", (key,)); self.db.commit()
        self.wake.set()

    def tombstone(self, key):
        valid_key(key)
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO deleted VALUES(?)", (key,))
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?)", (key, "", "", "trash", time.time()))
            self.db.execute("DELETE FROM objects WHERE key=?", (key,)); self.db.commit()
        self.wake.set()

    def sync_once(self):
        with self.lock:
            job = self.db.execute("SELECT key,sha,snapshot,operation FROM jobs ORDER BY created LIMIT 1").fetchone()
        if not job: return False
        key, sha, snapshot, operation = job
        try:
            if operation == "trash":
                self.client.put_object(Bucket=self.bucket, Key=PREFIX+"trash/"+key,
                    Body=json.dumps({"deleted": True, "time": time.time()}).encode(), ContentType="application/json")
            else:
                content = Path(snapshot).read_bytes()
                if hashlib.sha256(content).hexdigest() != sha:
                    raise StorageError("The pending save failed verification; the original is kept.")
                content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
                # Every metadata revision has a separate cloud copy before the current copy changes.
                if key.endswith((".json", ".jsonl")):
                    self.client.put_object(Bucket=self.bucket, Key=PREFIX+"history/"+key+"/"+uuid.uuid4().hex,
                        Body=content, Metadata={"sha256": sha}, ContentType=content_type)
                self.client.put_object(Bucket=self.bucket, Key=PREFIX+key, Body=content,
                    Metadata={"sha256": sha}, ContentType=content_type)
                head = self.client.head_object(Bucket=self.bucket, Key=PREFIX+key)
                if head.get("Metadata", {}).get("sha256") != sha or head["ContentLength"] != len(content):
                    raise StorageError("The cloud save did not verify; the original is kept.")
                self.client.put_object(Bucket=self.bucket, Key=PREFIX+"trash/"+key,
                    Body=b'{"deleted":false}', ContentType="application/json")
            with self.lock:
                self.db.execute("DELETE FROM jobs WHERE key=? AND sha=? AND operation=?", (key, sha, operation))
                if operation == "put":
                    self.db.execute("UPDATE objects SET verified=? WHERE key=? AND sha=?", (sha, key, sha))
                self.db.commit(); self.error = ""
            self._clean_snapshots(); self.prune()
            return True
        except Exception:
            self.error = "Cloud save pending. Check your connection or R2 access; local copies are kept."
            return False

    def _clean_snapshots(self):
        with self.lock:
            live = {r[0] for r in self.db.execute("SELECT snapshot FROM jobs")}
            for file in self.pending.glob("*.blob"):
                if str(file) not in live: file.unlink(missing_ok=True)

    def start(self):
        with self.lock:
            if self.worker and self.worker.is_alive(): return
            self.worker = Thread(target=self._run, daemon=True, name="lyric-goat-cloud")
            self.worker.start()

    def _run(self):
        while not self.closed:
            self.wake.clear()
            while not self.closed and self.sync_once(): pass
            self.wake.wait(30)

    @contextmanager
    def pin(self, path):
        name = str(Path(path).resolve())
        with self.lock: self.pins[name] = self.pins.get(name, 0) + 1
        try: yield
        finally:
            with self.lock:
                self.pins[name] -= 1
                if not self.pins[name]: del self.pins[name]

    def ensure(self, path, key):
        valid_key(key); path = Path(path).resolve()
        with self.lock:
            if self.db.execute("SELECT 1 FROM deleted WHERE key=?", (key,)).fetchone(): return None
            if path.exists():
                self.db.execute("UPDATE objects SET touched=? WHERE key=?", (time.time(), key)); self.db.commit()
                return path
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=PREFIX+key)
            try:
                if response.get("ContentLength", MAX_FILE+1) > MAX_FILE:
                    raise StorageError("The saved file is too large.")
                content = response["Body"].read(MAX_FILE+1)
            finally: response["Body"].close()
            sha = hashlib.sha256(content).hexdigest()
            if len(content) > MAX_FILE or response.get("Metadata", {}).get("sha256") != sha:
                raise StorageError("The downloaded take failed verification.")
            with self.lock:
                # Never replace a save that happened while this download was running.
                if path.exists(): return path
                if self.db.execute("SELECT 1 FROM deleted WHERE key=?", (key,)).fetchone(): return None
                atomic_bytes(path, content)
                self.db.execute("INSERT OR REPLACE INTO objects VALUES(?,?,?,?,?)", (key, str(path), sha, sha, time.time())); self.db.commit()
            return path
        except StorageError: raise
        except Exception:
            raise StorageError("Could not open this cloud file. Check your internet connection; your cloud copy is kept.") from None

    def prune(self, clear=False):
        removed = 0
        with self.lock:
            rows = self.db.execute("SELECT key,path,sha,verified FROM objects ORDER BY touched").fetchall()
            audio = [(k,Path(p),sha,verified) for k,p,sha,verified in rows if k.startswith("audio/") or k.endswith("/take.wav")]
            total = sum(p.stat().st_size for _,p,_,_ in audio if p.exists())
            target = 0 if clear else self.cache_bytes
            for key,path,sha,verified in audio:
                if total <= target: break
                if (sha != verified or str(path) in self.pins or not path.exists()
                        or self.db.execute("SELECT 1 FROM jobs WHERE key=?", (key,)).fetchone()): continue
                if digest_file(path) != verified: continue
                size = path.stat().st_size; path.unlink(); total -= size; removed += size
        return removed

    def status(self):
        with self.lock:
            pending = self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            paths = [Path(r[0]) for r in self.db.execute("SELECT path FROM objects")]
        return {"enabled": True, "pending": pending, "error": self.error,
            "cache_bytes": sum(p.stat().st_size for p in paths if p.exists()), "cache_limit": self.cache_bytes,
            "bucket": self.bucket, "state": "pending" if pending else "saved"}

    def restore_catalog(self):
        """Restore missing metadata only. Existing edits and pending local work win."""
        restored = 0
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            keys = [row["Key"] for page in paginator.paginate(Bucket=self.bucket, Prefix=PREFIX)
                    for row in page.get("Contents", [])]
            for full in keys:
                key = full.removeprefix(PREFIX)
                try: valid_key(key)
                except StorageError: continue
                if not key.endswith((".json", ".jsonl")): continue
                path = self.root / key
                if path.exists(): continue
                tombstone_key = PREFIX+"trash/"+key
                if tombstone_key in keys:
                    response = self.client.get_object(Bucket=self.bucket, Key=tombstone_key)
                    try: deleted = json.loads(response["Body"].read()).get("deleted")
                    finally: response["Body"].close()
                    if deleted: continue
                if self.ensure(path, key): restored += 1
            return restored
        except Exception:
            raise StorageError("Could not load the cloud library. Your existing local library is kept.") from None


def library(start=True):
    if not configured(): return None
    root = data_root(); identity = os.getenv("R2_ACCOUNT_ID") + "/" + os.getenv("R2_BUCKET")
    namespace = hashlib.sha256(identity.encode()).hexdigest()[:20]
    # Credentials only participate in an in-memory fingerprint; never put them in state paths.
    fingerprint = hashlib.sha256((identity+os.getenv("R2_ACCESS_KEY_ID")+os.getenv("R2_SECRET_ACCESS_KEY")).encode()).hexdigest()
    with _instances_lock:
        key = (str(root), namespace)
        if key not in _instances:
            for (old_root, _), old in _instances.items():
                if old_root == str(root):
                    old.closed = True
                    old.wake.set()
            _instances[key] = Library(root, make_client(), os.environ["R2_BUCKET"], namespace)
            _instances[key].credential_fingerprint = fingerprint
        instance = _instances[key]
        if getattr(instance, "credential_fingerprint", None) != fingerprint:
            # Reuse one outbox worker per destination, including when credentials rotate.
            instance.client = make_client()
            instance.credential_fingerprint = fingerprint
        instance.closed = False
    if start: instance.start()
    return instance


def queue_file(path, key):
    instance = library()
    if instance: instance.enqueue(path, key)


def deleted(key):
    instance = library()
    if instance: instance.tombstone(key)


def ensure_file(path, key):
    instance = library()
    return instance.ensure(path, key) if instance else (Path(path) if Path(path).exists() else None)


def audio_response(path, mimetype=None, key=None):
    from flask import send_file
    from werkzeug.wsgi import ClosingIterator
    lease = pinned(path)
    lease.__enter__()
    released = False
    def release():
        nonlocal released
        if not released:
            released = True
            lease.__exit__(None, None, None)
    try:
        if key and ensure_file(path, key) is None:
            raise StorageError("The saved audio could not be opened.")
        response = send_file(Path(path).resolve(), mimetype=mimetype, conditional=True)
        # send_file uses direct_passthrough: wrap its iterable so WSGI close also releases the pin.
        response.response = ClosingIterator(response.response, release)
        return response
    except Exception:
        release()
        raise


@contextmanager
def pinned(path):
    instance = library()
    if instance:
        with instance.pin(path): yield
    else: yield


def migrate_saved():
    instance = library()
    if not instance: return 0
    count = 0
    for path in instance.root.rglob("*"):
        if not path.is_file() or ".cloud" in path.relative_to(instance.root).parts: continue
        key = path.relative_to(instance.root).as_posix()
        try: valid_key(key)
        except StorageError: continue
        instance.enqueue(path, key); count += 1
    return count


def status():
    instance = library()
    return instance.status() if instance else {"enabled": False, "state": "local", "pending": 0, "error": ""}


def bootstrap():
    """Reconcile saves interrupted before enqueueing, without delaying studio startup."""
    try:
        if configured(): migrate_saved()
    except Exception:
        try:
            instance = library()
            if instance: instance.error = "Some files still need a cloud save. Use Retry pending saves; local copies are kept."
        except Exception: pass
