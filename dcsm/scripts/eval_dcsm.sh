#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

EXPERIMENT_DIR="${1:-}"
TAU="${2:-}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Usage: $0 <path_to_experiment_dir> [tau]"
    exit 1
fi

echo "Running DCSM evaluation for experiment in $EXPERIMENT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

EXTRA_ARGS=()
if [ -n "$TAU" ]; then
    EXTRA_ARGS+=(--tau "$TAU")
fi

python -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes=${NUM_PROCESSES} \
    scripts/eval_dcsm.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    "${EXTRA_ARGS[@]}"
