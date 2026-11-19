#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/train_dcsm_supcon.yaml}"
CHECKPOINT="${2:-checkpoints/encoder_epoch_100.pt}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

echo "Running DCSM training using encoder checkpoint: $CHECKPOINT"
echo "Using config: $CONFIG"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes=${NUM_PROCESSES} \
    scripts/train_dcsm.py \
    --config "$CONFIG" \
    --encoder_ckpt "$CHECKPOINT"

