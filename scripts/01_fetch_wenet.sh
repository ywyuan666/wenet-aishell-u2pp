#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh
ensure_gpu_env

if [ -d "$WENET_ROOT/.git" ]; then
  echo "WeNet already exists: $WENET_ROOT"
  cd "$WENET_ROOT"
  git fetch --depth 1 origin "$WENET_REF" || true
  git checkout "$WENET_REF" || true
else
  git clone --depth 1 --branch "$WENET_REF" https://github.com/wenet-e2e/wenet.git "$WENET_ROOT"
fi

cd "$WENET_ROOT"
echo "WeNet commit:"
git rev-parse --short HEAD

echo "== Install WeNet without overriding torch/torchaudio =="
REQ_NO_TORCH=/tmp/wenet_requirements_no_torch.txt
grep -vE '^(torch|torchaudio|deepspeed)([=<> ]|$)' requirements.txt > "$REQ_NO_TORCH" || cp requirements.txt "$REQ_NO_TORCH"
python -m pip install -r "$REQ_NO_TORCH" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e . --no-deps

echo "Done. Next: bash scripts/02_prepare_aishell.sh"