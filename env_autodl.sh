#!/usr/bin/env bash
set -euo pipefail

# ── 自动检测项目根目录（env.sh 所在目录） ──
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ── 平台检测 ──
IS_WINDOWS=false
IS_AUTODL=false
case "$(uname -s 2>/dev/null || echo 'unknown')" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
  Linux)                IS_WINDOWS=false ;;
esac

if [ -d "/root/autodl-tmp" ]; then
  IS_AUTODL=true
fi

# ── 数据根目录 ──
if $IS_AUTODL; then
  export AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
elif $IS_WINDOWS; then
  export AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-$PROJECT_ROOT}"
else
  export AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-$PROJECT_ROOT}"
fi

export WENET_ROOT="${WENET_ROOT:-$AUTODL_DATA_ROOT/wenet}"
export AISHELL_ROOT="${AISHELL_ROOT:-$AUTODL_DATA_ROOT/datasets/aishell}"
export RUNTIME_MODEL_ROOT="${RUNTIME_MODEL_ROOT:-$AUTODL_DATA_ROOT/wenet_runtime_models}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

# ── 缓存路径 ──
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$AUTODL_DATA_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$AUTODL_DATA_ROOT/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$AUTODL_DATA_ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$AUTODL_DATA_ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$AUTODL_DATA_ROOT/.cache/modelscope}"

# ── 训练 / 数据参数 ──
export WENET_REF="${WENET_REF:-main}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-conf/train_u2++_conformer.yaml}"
export EXP_DIR="${EXP_DIR:-exp/u2pp_conformer_course}"
export TRAIN_SET="${TRAIN_SET:-train_subset}"
export DEV_SET="${DEV_SET:-dev}"
export DATA_TYPE="${DATA_TYPE:-raw}"
export TRAIN_ENGINE="${TRAIN_ENGINE:-torch_ddp}"
export AVERAGE_NUM="${AVERAGE_NUM:-10}"

# ── CPU/GPU 自适应 ──
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export CUDA_AVAILABLE=true
  export NUM_WORKERS="${NUM_WORKERS:-8}"
  export NJ="${NJ:-16}"
  export CUDA_DEVICE="${CUDA_DEVICE:-0}"
else
  export CUDA_AVAILABLE=false
  export NUM_WORKERS="${NUM_WORKERS:-2}"
  export NJ="${NJ:-2}"
  export CUDA_DEVICE=""
  echo ">>> CPU-only mode detected (no CUDA GPU). Training will be slow but functional. <<<"
fi

# ── 子集大小（CPU 模式自动减半） ──
if $CUDA_AVAILABLE; then
  export SUBSET_UTTS="${SUBSET_UTTS:-3000}"
else
  export SUBSET_UTTS="${SUBSET_UTTS:-100}"
  export TRAIN_CONFIG="${TRAIN_CONFIG:-conf/train_u2++_conformer.yaml}"
  # 减小 batch 适配 CPU
  export CPU_BATCH="${CPU_BATCH:-2}"
fi

# ── 数据 archive 路径 ──
export DATA_ARCHIVE="$DATA_DIR/data_aishell.tgz"
export RESOURCE_ARCHIVE="$DATA_DIR/resource_aishell.tgz"

# ── 创建必要目录 ──
mkdir -p "$AUTODL_DATA_ROOT" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$MODELSCOPE_CACHE" "$RUNTIME_MODEL_ROOT" "$AISHELL_ROOT"

# ── 脚本级共享函数：source 后自动启用 GPU ──
ensure_gpu_env() {
    if $CUDA_AVAILABLE; then
        export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    fi
}

echo "=== Environment Loaded ==="
echo "PROJECT_ROOT     = $PROJECT_ROOT"
echo "AUTODL_DATA_ROOT = $AUTODL_DATA_ROOT"
echo "WENET_ROOT       = $WENET_ROOT"
echo "AISHELL_ROOT     = $AISHELL_ROOT"
echo "CUDA_AVAILABLE   = $CUDA_AVAILABLE"
echo "IS_WINDOWS       = $IS_WINDOWS"
echo "IS_AUTODL        = $IS_AUTODL"
echo "SUBSET_UTTS      = $SUBSET_UTTS"
echo "========================="
