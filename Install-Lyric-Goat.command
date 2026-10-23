#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! /bin/bash scripts/install_macos.sh; then
  read -r -p "Setup stopped. Press Return to close. "
  exit 1
fi
