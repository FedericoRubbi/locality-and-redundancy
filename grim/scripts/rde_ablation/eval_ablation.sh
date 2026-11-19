#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <method_name> [--jobs N] [--eval-config PATH] [additional_args...]"
    echo ""
    echo "Available methods:"
    ls scripts/rde_ablation/methods/*.py | sed 's|^scripts/rde_ablation/methods/||' | sed 's|\.py$|  |'
    exit 1
fi

METHOD="$1"
shift

MAX_JOBS=1
EVAL_CONFIG="configs/ablations/eval.yaml"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --jobs|-j)
            MAX_JOBS="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ "$MAX_JOBS" -lt 1 ] 2>/dev/null; then
    MAX_JOBS=1
fi

run_one() {
    local method="$1"
    local port="$2"
    shift 2

    local script="scripts/rde_ablation/methods/${method}.py"
    local config="configs/ablations/${method}.yaml"

    if [ ! -f "$script" ]; then
        echo "Error: Cannot find script '$script'"
        return 1
    fi

    if [ ! -f "$config" ]; then
        echo "Error: Cannot find config '$config'"
        return 1
    fi

    echo "Starting Ablation Evaluation: ${method}"
    accelerate launch --main_process_port "$port" "$script" --config "$config" --run-name "$method" --eval-config "$EVAL_CONFIG" --task eval "$@"
}

if [ "$METHOD" = "all" ]; then
    mapfile -t methods < <(find scripts/rde_ablation/methods -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sed 's/\.py$//' | sort)

    if [ "${#methods[@]}" -eq 0 ]; then
        echo "No ablation scripts found"
        exit 1
    fi

    log_dir="logs/ablations/eval_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_dir"

    echo "Starting all evaluations with up to ${MAX_JOBS} concurrent runs (${#methods[@]} total)."
    echo "Logs directory: $log_dir"
    echo "============================================================"

    base_port=29500
    failed=0
    pids=()
    names=()
    logs=()
    
    for i in "${!methods[@]}"; do
        method="${methods[$i]}"
        log_file="$log_dir/${method}.log"
        port=$((base_port + i))

        while [ "${#pids[@]}" -ge "$MAX_JOBS" ]; do
            if wait "${pids[0]}"; then
                echo "[OK] ${names[0]}"
            else
                echo "[FAIL] ${names[0]} (see ${logs[0]})"
                failed=1
            fi
            pids=("${pids[@]:1}")
            names=("${names[@]:1}")
            logs=("${logs[@]:1}")
        done
        
        echo "Evaluating ${method} -> $log_file"
        (
            run_one "$method" "$port" "$@"
        ) >"$log_file" 2>&1 &
        pids+=("$!")
        names+=("$method")
        logs+=("$log_file")
    done

    while [ "${#pids[@]}" -gt 0 ]; do
        if wait "${pids[0]}"; then
            echo "[OK] ${names[0]}"
        else
            echo "[FAIL] ${names[0]} (see ${logs[0]})"
            failed=1
        fi
        pids=("${pids[@]:1}")
        names=("${names[@]:1}")
        logs=("${logs[@]:1}")
    done

    echo "============================================================"
    if [ "$failed" -eq 0 ]; then
        echo "All evaluations completed successfully."
        exit 0
    else
        echo "Some evaluations failed. Check logs in $log_dir"
        exit 1
    fi
fi

SCRIPT="scripts/rde_ablation/methods/${METHOD}.py"
CONFIG="configs/ablations/${METHOD}.yaml"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Cannot find script '$SCRIPT'"
    exit 1
fi

accelerate launch --main_process_port 29500 "$SCRIPT" --config "$CONFIG" --run-name "$METHOD" --eval-config "$EVAL_CONFIG" --task eval "$@"
