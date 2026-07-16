#!/usr/bin/env bash
set -euo pipefail

bash scripts/00_prepare_autodl.sh
bash scripts/01_fetch_wenet.sh
bash scripts/02_prepare_aishell.sh
bash scripts/03_train_course_fast.sh
bash scripts/04_decode_eval.sh
bash scripts/05_export_model.sh
bash scripts/06_package_runtime_model.sh