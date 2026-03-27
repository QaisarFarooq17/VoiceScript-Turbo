#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_transcribe.sh /path/to/audio.wav [language]"
  exit 1
fi

INPUT="$1"
LANGUAGE="${2:-en}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/qaisar/miniconda3/envs/Finn_Plus/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at: $PYTHON_BIN"
  echo "Update PYTHON_BIN in scripts/run_transcribe.sh if you want another env."
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT"
  exit 1
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.transcribe \
  --input "$INPUT" \
  --output outputs/transcript.txt \
  --json-output outputs/transcript.segments.json \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language "$LANGUAGE" \
  --vad
