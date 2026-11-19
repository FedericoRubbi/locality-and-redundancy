#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for s in itmf vit dcsm grim fedsym; do
  "$PY" scripts/figures_$s.py
done
