#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/qaisar/miniconda3/envs/Finn_Plus/bin/python"
LANGUAGE="en"
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    EXTRA_ARGS=("$@")
  else
    LANGUAGE="$1"
    shift
    EXTRA_ARGS=("$@")
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at: $PYTHON_BIN"
  echo "Update PYTHON_BIN in scripts/run_live_mic.sh if you want another env."
  exit 1
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.live_mic \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language "$LANGUAGE" \
  --chunk-seconds 3 \
  --beam-size 1 \
  --vad \
  --output outputs/live_transcript.txt \
  "${EXTRA_ARGS[@]}"
