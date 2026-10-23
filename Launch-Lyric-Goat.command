#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! /bin/bash scripts/launch_macos.sh; then
  read -r -p "Launch stopped. Press Return to close. "
  exit 1
fi
