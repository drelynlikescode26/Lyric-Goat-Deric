#!/bin/bash
set -e
cd "$(dirname "$0")"
export LYRIC_GOAT_OPEN_PATH=/compare
exec /bin/bash Launch-Lyric-Goat.command
