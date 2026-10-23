#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is for macOS. Use python scripts/run_local.py on Linux."
  exit 1
fi
command -v ffmpeg >/dev/null 2>&1 || { echo "Run Install-Lyric-Goat.command to install ffmpeg."; exit 1; }
if [[ ! -d .git ]]; then
  echo "Run Install-Lyric-Goat.command once to create the updateable installation."
  exit 1
fi
if [[ "$(git branch --show-current)" == "main" && -z "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Checking GitHub for updates…"
  if git -c credential.interactive=false fetch origin main; then
    if ! git merge --ff-only origin/main; then
      echo "Local history differs from GitHub. Files were preserved; running the existing version."
    fi
  else
    echo "GitHub could not be reached. Running the installed version."
  fi
else
  echo "Local code changes or another branch found. Automatic updates skipped."
fi
if command -v brew >/dev/null 2>&1 && [[ -x "$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12" ]]; then
  python_bin="$(brew --prefix python@3.12)/bin/python3.12"
else
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "$python_bin" ]] || ! "$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14)' 2>/dev/null; then
  echo "Run Install-Lyric-Goat.command to install a supported Python."
  exit 1
fi
if [[ ! -d .venv ]]; then "$python_bin" -m venv .venv; fi
if ! .venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14)' 2>/dev/null; then
  echo "The app's .venv needs repair. Your .env and songs were left untouched."
  exit 1
fi
requirements_hash="$(.venv/bin/python -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())')"
installed_hash="$(cat .venv/requirements.sha256 2>/dev/null || true)"
if [[ "$requirements_hash" != "$installed_hash" ]]; then
  echo "Preparing Lyric Goat dependencies. The first run can take a few minutes…"
  .venv/bin/python -m pip install -r requirements.txt
  printf '%s' "$requirements_hash" > .venv/requirements.sha256
fi
exec .venv/bin/python scripts/run_local.py
