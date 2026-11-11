#!/usr/bin/env bash
# End-to-end pipeline: install deps -> download data -> fine-tune -> sample translations.
#
# Usage (from anywhere; works locally or in Colab via `!bash scripts/run_pipeline.sh`):
#   bash scripts/run_pipeline.sh                      # full QLoRA run with defaults
#   bash scripts/run_pipeline.sh --num_epochs 1       # extra args are passed to train.py
#   SKIP_INSTALL=1 bash scripts/run_pipeline.sh       # skip `pip install`
#
# Requires a CUDA GPU for 4-bit training (a free Colab T4 is enough).
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "==> [1/4] Installing dependencies"
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -q -r requirements.txt
fi

echo "==> [2/4] Preparing data"
if [[ ! -f data/hinge_train.json ]]; then
  "$PYTHON" utils/prepare_training_data.py --dataset hinge
else
  echo "data/hinge_train.json already present, skipping download"
fi

echo "==> [3/4] Fine-tuning"
"$PYTHON" train.py "$@"

echo "==> [4/4] Sample translations (base model vs fine-tuned)"
"$PYTHON" infer.py --compare \
  --text "Where are you going tonight?" \
  --text "I will call you after the meeting." \
  --text "This movie was much better than I expected."
