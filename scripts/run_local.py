"""Private local runner; no debug server or public network binding."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def ready(url):
    try:
        with urlopen(url + "/health", timeout=1) as response:
            return json.load(response).get("app") == "lyric-goat"
    except (URLError, ValueError, OSError):
        return False


def open_studio(url):
    for _ in range(120):
        if ready(url):
            destination = url if os.getenv("OPENAI_API_KEY") else url + "/settings"
            page = os.getenv("LYRIC_GOAT_OPEN_PATH")
            if page in {"/compare", "/settings", "/"}:
                destination = url + page
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Safari", destination], check=False)
            else:
                print("Open " + destination, flush=True)
            return
        time.sleep(.5)
    print("Startup is taking longer than expected. Check this window for errors.", flush=True)


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Restrict newly created settings, recordings and song files to this user.
    os.umask(0o077)
    os.environ["LYRIC_GOAT_LOCAL_SETUP"] = "1"
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    port = int(os.getenv("LYRIC_GOAT_PORT", "8765"))
    if not 1024 <= port <= 65535:
        raise SystemExit("LYRIC_GOAT_PORT must be between 1024 and 65535.")
    url = f"http://127.0.0.1:{port}"
    runtime = ROOT / ".local-runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / "server.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if ready(url):
                print("Lyric Goat is already open. Close its Terminal window and reopen to load code updates.")
                open_studio(url)
                return
            raise SystemExit("Another Lyric Goat instance is starting. Try again shortly.")
        # Identify port collisions without opening an unrelated local application.
        with socket.socket() as probe:
            # Match the server socket so a recent stop does not look like a port collision.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"Port {port} is in use. Close the other app or change LYRIC_GOAT_PORT in .env.")
        from app import app
        from services.cloud_storage import bootstrap
        from waitress import serve
        print(f"Lyric Goat: {url}\nLeave this Terminal window open. Press Control-C to stop.\n"
              "Close it before reopening the Desktop launcher to load updates.", flush=True)
        Thread(target=open_studio, args=(url,), daemon=True).start()
        Thread(target=bootstrap, daemon=True).start()
        serve(app, host="127.0.0.1", port=port, threads=4,
              max_request_body_size=50 * 1024 * 1024, channel_timeout=300)


if __name__ == "__main__":
    main()
