#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_HOME="${SATURNI_HOME:-$DATA_HOME/saturni-rag}"
VENV_DIR="$APP_HOME/venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
PULL_MODELS=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [--pull-models]

Installs Saturni into an isolated virtual environment and creates:
  ~/.local/bin/saturni
  ~/.local/bin/saturni-rag

Options:
  --pull-models  Pull nomic-embed-text and gemma2:2b after installation
  -h, --help     Show this help
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --pull-models) PULL_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

find_python() {
  local candidate
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Saturni requires Python 3.10 or newer." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/pyproject.toml" ]]; then
  echo "Run this installer from the Saturni repository." >&2
  exit 1
fi

mkdir -p "$APP_HOME" "$BIN_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "Unable to create a virtual environment." >&2
    echo "On Ubuntu/Pop!_OS, install it with: sudo apt install python3-venv" >&2
    exit 1
  fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --upgrade "$ROOT_DIR"

ln -sfn "$VENV_DIR/bin/saturni" "$BIN_DIR/saturni"
ln -sfn "$VENV_DIR/bin/saturni-rag" "$BIN_DIR/saturni-rag"
install -m 0755 "$ROOT_DIR/uninstall.sh" "$APP_HOME/uninstall.sh"

"$BIN_DIR/saturni" --version

if (( PULL_MODELS )); then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed. Install it, start 'ollama serve', then run:" >&2
    echo "  saturni pull nomic-embed-text" >&2
    echo "  saturni pull gemma2:2b" >&2
  else
    ollama pull nomic-embed-text
    ollama pull gemma2:2b
  fi
fi

cat <<EOF

Saturni installed successfully.

Command:  $BIN_DIR/saturni
Data:     $APP_HOME/data
Doctor:   saturni doctor

If 'saturni' is not found in a new shell, add this to ~/.bashrc or ~/.zshrc:
  export PATH="$BIN_DIR:\$PATH"

EOF
