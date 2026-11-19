#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
LOG=${LOG:-logs}
N=8
mkdir -p "$LOG"
for phase in search final; do
  echo "$(date '+%F %T') phase $phase"
  for i in $(seq 0 $((N - 1))); do
    "$PY" rerun.py --phase "$phase" --shard "$i" --num-shards "$N" --device "cuda:$((i % 2))" \
      > "$LOG/${phase}_$i.log" 2>&1 &
  done
  wait
done
grep -l "Error\|Traceback" "$LOG"/*.log || true
