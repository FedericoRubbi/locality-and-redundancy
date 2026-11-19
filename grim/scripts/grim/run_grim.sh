#!/bin/bash
echo "=== GRIM Training & Evaluation ==="
python scripts/grim/train_grim.py --config configs/grim/train.yaml --eval_config configs/grim/eval.yaml
