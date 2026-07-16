#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

cd "$WENET_ROOT/examples/aishell/s0"

echo "== Prepare AISHELL-1 data =="
echo "AISHELL_ROOT=$AISHELL_ROOT"

# 检查数据是否已存在
if [ -d "$AISHELL_ROOT/data_aishell/wav" ] && [ -d "$AISHELL_ROOT/resource_aishell" ]; then
  echo ">>> AISHELL data already exists at $AISHELL_ROOT. Skipping download stage."
  STAGE=1  # 跳过下载，直接从数据格式化开始
else
  echo ">>> AISHELL data not found. Attempting download via run.sh --stage -1 ..."
  STAGE=-1
fi

bash run.sh \
  --stage $STAGE \
  --stop_stage 3 \
  --data "$AISHELL_ROOT" \
  --data_type "$DATA_TYPE" \
  --nj "$NJ" \
  --num_workers "$NUM_WORKERS" \
  --train_config "$TRAIN_CONFIG" \
  --dir "$EXP_DIR"

echo "Done. Next: bash scripts/03_train_course_fast.sh"
