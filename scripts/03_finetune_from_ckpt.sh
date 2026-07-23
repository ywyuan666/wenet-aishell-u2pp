#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/avg.pt or final.pt before running fine-tune.}"
export TRAIN_SET="${TRAIN_SET:-my_train}"
export EXP_DIR="${EXP_DIR:-exp/finetune}"

cd "$WENET_ROOT/examples/aishell/s0"

echo "== Fine-tune from checkpoint =="
echo "CHECKPOINT=$CHECKPOINT"
echo "TRAIN_SET=$TRAIN_SET"
echo "EXP_DIR=$EXP_DIR"

# GPU/CPU 自适应
if $CUDA_AVAILABLE; then
  export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
fi
bash run.sh \
  --stage 4 --stop_stage 4 --data "$AISHELL_ROOT" --data_type "$DATA_TYPE" \
  --nj "$NJ" --num_workers "$NUM_WORKERS" --train_config "$TRAIN_CONFIG" \
  --train_set "$TRAIN_SET" --dir "$EXP_DIR" --train_engine "$TRAIN_ENGINE" \
  --checkpoint "$CHECKPOINT"

echo "Done. Next: EXP_DIR=$EXP_DIR bash scripts/04_decode_eval.sh"
