#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

python "$PROJECT_ROOT/tools/summarize_wenet_results.py" \
  --exp "$WENET_ROOT/examples/aishell/s0/$EXP_DIR" \
  --out "$PROJECT_ROOT/results_summary.md"

cat "$PROJECT_ROOT/results_summary.md"