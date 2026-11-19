#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

CHECKPOINT=${1:-"checkpoints/encoder_epoch_100.pt"}

echo "Running linear probing on checkpoint: $CHECKPOINT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -m accelerate.commands.launch --multi_gpu --num_processes=2 scripts/linear_probe.py --config configs/pretrain_cifar100.yaml --checkpoint "$CHECKPOINT" --epochs 30 --lr 1e-3 --batch_size 512
