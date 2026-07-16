#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

echo "== System Info =="
if command -v uname >/dev/null 2>&1; then
  uname -a
fi
df -h . || true

echo "== CUDA / GPU =="
if $CUDA_AVAILABLE; then
  nvidia-smi || true
else
  echo "CPU-only mode. Skipping GPU checks."
fi

echo "== Python =="
python -V
python -m pip -V

python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    import sys
    print("python:", sys.version)
except Exception as exc:
    print("torch check failed:", repr(exc))
PY

echo "== Install common Python packages =="
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
python -m pip install -U pyyaml tqdm requests pandas editdistance soundfile sentencepiece jieba kaldiio textgrid tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet

echo "Done. Next: bash scripts/01_fetch_wenet.sh"
