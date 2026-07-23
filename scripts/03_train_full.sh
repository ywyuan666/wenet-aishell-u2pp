#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh
ensure_gpu_env

cd "$WENET_ROOT/examples/aishell/s0"
export TRAIN_SET="${TRAIN_SET:-train}"
export EXP_DIR="${EXP_DIR:-exp/u2pp_conformer_full}"

echo "== Full AISHELL training =="
echo "TRAIN_SET=$TRAIN_SET"
echo "EXP_DIR=$EXP_DIR"

bash run.sh \
  --stage 4 --stop_stage 4 --data "$AISHELL_ROOT" --data_type "$DATA_TYPE" \
  --nj "$NJ" --num_workers "$NUM_WORKERS" --train_config "$TRAIN_CONFIG" \
  --train_set "$TRAIN_SET" --dir "$EXP_DIR" --train_engine "$TRAIN_ENGINE"

echo "Done. Next: EXP_DIR=$EXP_DIR bash scripts/04_decode_eval.sh"
