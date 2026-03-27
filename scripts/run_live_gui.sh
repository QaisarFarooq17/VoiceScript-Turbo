#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/qaisar/miniconda3/envs/Finn_Plus/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at: $PYTHON_BIN"
  echo "Update PYTHON_BIN in scripts/run_live_gui.sh if you want another env."
  exit 1
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.live_gui \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language en \
  --chunk-seconds 3 \
  --beam-size 1 \
  --vad \
  "$@"
