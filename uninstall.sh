#!/usr/bin/env bash
set -Eeuo pipefail

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_HOME="${SATURNI_HOME:-$DATA_HOME/saturni-rag}"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
PURGE=0

if [[ "${1:-}" == "--purge" ]]; then
  PURGE=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--purge]" >&2
  exit 2
fi

rm -f "$BIN_DIR/saturni" "$BIN_DIR/saturni-rag"
rm -rf "$APP_HOME/venv"

if (( PURGE )); then
  rm -rf "$APP_HOME"
  echo "Saturni and all indexed data were removed."
else
  echo "Saturni was removed. Indexed data remains in: $APP_HOME/data"
  echo "Run '$0 --purge' to remove the data too."
fi
