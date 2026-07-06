#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Edit this, then just run ./cli.sh ----
# Uses json/<NAME>.json + input/<NAME>.csv (if it exists; otherwise falls
# back to that json's own "input" field), and writes results to output/<NAME>/
NAME="config"
# ---------------------------------------------

PYTHON_BIN="$HOME/miniconda3/envs/scarabinfra/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/cli.py" --name "$NAME" "$@"
