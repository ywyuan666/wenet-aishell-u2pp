#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

cd "$WENET_ROOT/examples/aishell/s0"

echo "== Build subset data.list =="
python "$PROJECT_ROOT/tools/make_subset_data_list.py" \
  --input data/train/data.list \
  --output "data/$TRAIN_SET/data.list" \
  --num "$SUBSET_UTTS"

echo "== Fast course training =="
echo "TRAIN_SET=$TRAIN_SET"
echo "EXP_DIR=$EXP_DIR"
echo "CUDA_AVAILABLE=$CUDA_AVAILABLE"
echo "SUBSET_UTTS=$SUBSET_UTTS"

# GPU/CPU 自适应：统一 run.sh 调用
if $CUDA_AVAILABLE; then
  export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
fi
bash run.sh \
  --stage 4 \
  --stop_stage 4 \
  --data "$AISHELL_ROOT" \
  --data_type "$DATA_TYPE" \
  --nj "$NJ" \
  --num_workers "$NUM_WORKERS" \
  --train_config "$TRAIN_CONFIG" \
  --train_set "$TRAIN_SET" \
  --dir "$EXP_DIR" \
  --train_engine "$TRAIN_ENGINE"

echo "Done. Next: bash scripts/04_decode_eval.sh"
