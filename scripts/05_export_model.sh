#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

cd "$WENET_ROOT/examples/aishell/s0"

echo "== Export JIT / final.zip =="
echo "EXP_DIR=$EXP_DIR"

bash run.sh \
  --stage 6 --stop_stage 6 --data "$AISHELL_ROOT" --data_type "$DATA_TYPE" \
  --train_config "$TRAIN_CONFIG" --dir "$EXP_DIR" --average_num "$AVERAGE_NUM"

echo "Search exported files:"
find "$EXP_DIR" -maxdepth 4 -type f \( -name 'final.zip' -o -name '*.jit' -o -name '*.pt' \) | sort

echo "Done. Next: bash scripts/06_package_runtime_model.sh"
