#!/usr/bin/env bash
# Install the desktop toggle for the user running this script.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
chmod +x "$ROOT/nomad-turbo"
mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/nomad-turbo" "$HOME/.local/bin/nomad-turbo"
exec "$ROOT/nomad-turbo" install
