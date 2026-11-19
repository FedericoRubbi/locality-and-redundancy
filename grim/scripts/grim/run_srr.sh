#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate

gpu=0
for cfg in "$@"; do
    tag=$(basename "$cfg" .yaml); tag=${tag#train_}
    log="logs/${tag}.log"
    name=$(grep -m1 '^run_name:' "$cfg" | sed 's/.*"\(.*\)".*/\1/')
    state="checkpoints/grim/${name}/train_state.pt"

    if pgrep -f "train_srr.py --config $cfg" > /dev/null; then
        echo "[resume] $name already running, skipping"
    elif [ -f "$state" ] && python -c "
import sys, yaml, torch
epochs = yaml.safe_load(open('$cfg'))['training']['epochs']
done = torch.load('$state', map_location='cpu', weights_only=False)['epoch']
sys.exit(0 if done >= epochs else 1)
" 2>/dev/null; then
        echo "[resume] $name already finished, skipping"
    else
        [ -f "$log" ] && mv "$log" "${log%.log}_$(date +%Y%m%d_%H%M%S).log"
        echo "[resume] relaunching $name on GPU $gpu at $(date)"
        setsid nohup env CUDA_VISIBLE_DEVICES=$gpu python -u scripts/grim/train_srr.py \
            --config "$cfg" --resume > "$log" 2>&1 < /dev/null &
    fi
    gpu=$(( (gpu + 1) % 2 ))
done
