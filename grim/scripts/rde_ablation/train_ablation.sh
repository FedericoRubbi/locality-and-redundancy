#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <method_name> [additional_args...]"
    echo ""
    echo "Available methods:"
    ls scripts/rde_ablation/methods/*.py | sed 's|^scripts/rde_ablation/methods/||' | sed 's|\.py$|  |'
    exit 1
fi

METHOD="$1"
shift

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

    echo "Starting Ablation Training: ${method}"
    echo "Script: $script"
    echo "Config: $config"
    echo "Port: $port"
    echo "============================================================"
    accelerate launch --main_process_port "$port" "$script" --config "$config" --run-name "$method" --task train "$@"
}

if [ "$METHOD" = "all" ]; then
    mapfile -t methods < <(find scripts/rde_ablation/methods -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sed 's/\.py$//' | sort)

    if [ "${#methods[@]}" -eq 0 ]; then
        echo "No ablation scripts found in scripts/rde_ablation/methods"
        exit 1
    fi

    log_dir="logs/ablations/train_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_dir"

    echo "Starting all ablations in parallel (${#methods[@]} runs)."
    echo "Logs directory: $log_dir"
    echo "Use 'tail -f $log_dir/<method>.log' to monitor a specific run."
    echo "============================================================"

    pids=()
    names=()
    base_port=29500
    
    for i in "${!methods[@]}"; do
        method="${methods[$i]}"
        log_file="$log_dir/${method}.log"
        port=$((base_port + i))
        
        echo "Launching ${method} (port $port) -> $log_file"
        (
            run_one "$method" "$port" "$@"
        ) >"$log_file" 2>&1 &
        
        pids+=("$!")
        names+=("$method")
    done

    echo "============================================================"
    echo "All jobs launched. Waiting for completion..."

    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[OK] ${names[$i]}"
        else
            echo "[FAIL] ${names[$i]} (see $log_dir/${names[$i]}.log)"
            failed=1
        fi
    done

    echo "============================================================"
    if [ "$failed" -eq 0 ]; then
        echo "All training completed successfully."
        exit 0
    else
        echo "Some training failed. Check logs in $log_dir"
        exit 1
    fi
fi

SCRIPT="scripts/rde_ablation/methods/${METHOD}.py"
CONFIG="configs/ablations/${METHOD}.yaml"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Cannot find script '$SCRIPT'"
    exit 1
fi

accelerate launch --main_process_port 29500 "$SCRIPT" --config "$CONFIG" --run-name "$METHOD" --task train "$@"
