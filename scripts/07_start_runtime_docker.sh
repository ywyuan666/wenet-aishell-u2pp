#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env_autodl.sh

MODEL_DIR="${MODEL_DIR:-$RUNTIME_MODEL_ROOT/aishell_u2pp_conformer}"
PORT="${PORT:-10086}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed on this AutoDL image." >&2
  echo "The exported model package is still available in: $MODEL_DIR" >&2
  exit 1
fi

if [ ! -f "$MODEL_DIR/final.zip" ] || [ ! -f "$MODEL_DIR/units.txt" ]; then
  echo "Missing final.zip or units.txt in $MODEL_DIR. Run scripts/06_package_runtime_model.sh first." >&2
  exit 1
fi

echo "== Start WeNet runtime docker =="
echo "MODEL_DIR=$MODEL_DIR"
echo "PORT=$PORT"
echo "This uses the official wenet mini runtime image. If the upstream image command changes, open an interactive shell and follow runtime/libtorch README."

docker run --gpus all --rm -it \
  -p "$PORT:$PORT" \
  -v "$MODEL_DIR:/home/wenet/model" \
  wenetorg/wenet-mini:v2.0.2 \
  bash /home/run.sh