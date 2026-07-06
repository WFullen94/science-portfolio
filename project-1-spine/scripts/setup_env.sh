#!/usr/bin/env bash
# One-shot environment setup for Project 1 (The Spine).
# Creates a Python 3.12 venv, installs deps, and wires up JAVA_HOME for PySpark.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
VENV_DIR="${VENV_DIR:-$HERE/.venv}"

# --- JAVA_HOME (PySpark/Delta need a JDK) ------------------------------------
if [ -z "${JAVA_HOME:-}" ]; then
  for c in /opt/homebrew/opt/openjdk@17 /opt/homebrew/opt/openjdk; do
    if [ -d "$c" ]; then export JAVA_HOME="$c"; break; fi
  done
fi
echo "JAVA_HOME=${JAVA_HOME:-<unset — install: brew install openjdk@17>}"

# --- venv --------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "Then run the pipeline stages via the Makefile:  make ingest / features / train / serve"
