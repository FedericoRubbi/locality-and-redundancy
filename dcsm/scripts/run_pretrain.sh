#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

NUM_GPUS=2

echo "Starting distributed pretraining on ${NUM_GPUS} GPUs from ${PROJECT_ROOT}..."

python -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    scripts/pretrain_encoder.py
