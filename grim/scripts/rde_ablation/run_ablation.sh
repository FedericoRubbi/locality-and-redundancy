#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <method_name> [additional_args...] [--eval-jobs N] [--eval-config PATH]"
    exit 1
fi

METHOD="$1"
shift

TRAIN_ARGS=()
EVAL_ARGS=()
EVAL_JOBS=1
EVAL_CONFIG="configs/ablations/eval.yaml"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --eval-jobs|-j)
            EVAL_JOBS="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "--- PHASE 1: TRAINING ---"
bash scripts/rde_ablation/train_ablation.sh "$METHOD" "${TRAIN_ARGS[@]}"
TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "Training phase failed. Aborting evaluation."
    exit $TRAIN_STATUS
fi

echo ""
echo "--- PHASE 2: EVALUATION ---"
bash scripts/rde_ablation/eval_ablation.sh "$METHOD" --jobs "$EVAL_JOBS" --eval-config "$EVAL_CONFIG" "${EVAL_ARGS[@]}"
