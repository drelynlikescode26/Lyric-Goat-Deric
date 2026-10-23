#!/bin/bash
set -euo pipefail
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS. See README for other platforms."
  exit 1
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew from https://brew.sh, then reopen this installer."
  echo "Homebrew supplies Python and ffmpeg; it may need your Mac password."
  exit 1
fi
brew list --versions python@3.12 >/dev/null || brew install python@3.12
command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg
install_dir="${LYRIC_GOAT_INSTALL_DIR:-$HOME/Applications/Lyric-Goat}"
repo_url="https://github.com/drelynlikescode26/Lyric-Goat-Deric.git"
mkdir -p "$(dirname "$install_dir")"
if [[ ! -e "$install_dir" ]]; then
  git clone "$repo_url" "$install_dir"
elif [[ ! -d "$install_dir/.git" ]]; then
  echo "An existing folder is at $install_dir. Move it or choose LYRIC_GOAT_INSTALL_DIR."
  exit 1
elif [[ "$(git -C "$install_dir" remote get-url origin)" != "$repo_url" && "$(git -C "$install_dir" remote get-url origin)" != "${repo_url%.git}" ]]; then
  echo "The existing folder is a different repository. Setup stopped."
  exit 1
fi
mkdir -p "$HOME/Desktop"
desktop_launcher="$HOME/Desktop/Lyric-Goat.command"
if [[ -e "$desktop_launcher" ]] && ! /usr/bin/grep -q 'Lyric Goat desktop launcher' "$desktop_launcher"; then
  echo "An unrelated Desktop launcher already exists. It was left untouched."
else
  printf '#!/bin/bash\n# Lyric Goat desktop launcher\nexec /bin/bash %q\n' "$install_dir/Launch-Lyric-Goat.command" > "$desktop_launcher"
  chmod 755 "$desktop_launcher"
fi
echo "Installed at $install_dir. Open Lyric-Goat.command on your Desktop next time."
exec /bin/bash "$install_dir/Launch-Lyric-Goat.command"
