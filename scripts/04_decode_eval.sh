#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh
ensure_gpu_env

cd "$WENET_ROOT/examples/aishell/s0"

echo "== Decode and CER evaluation =="
echo "EXP_DIR=$EXP_DIR"
echo "AVERAGE_NUM=$AVERAGE_NUM"

bash run.sh \
  --stage 5 --stop_stage 5 --data "$AISHELL_ROOT" --data_type "$DATA_TYPE" \
  --nj "$NJ" --num_workers "$NUM_WORKERS" --train_config "$TRAIN_CONFIG" \
  --dir "$EXP_DIR" --average_num "$AVERAGE_NUM"

python "$PROJECT_ROOT/tools/summarize_wenet_results.py" \
  --exp "$WENET_ROOT/examples/aishell/s0/$EXP_DIR" \
  --out "$PROJECT_ROOT/results_summary.md"

echo "Done. Summary: $PROJECT_ROOT/results_summary.md"
