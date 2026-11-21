#!/usr/bin/env bash
# Run from any directory. Pass an empty output directory for a new reproduction.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$(realpath "$1")"
OUT="$(realpath -m "$2")"
PY="${COG_PYTHON:-python}"
cd "$HERE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
for advice in assisted unaided; do
  opts=()
  dest="$OUT"
  if [[ "$advice" == "unaided" ]]; then
    opts+=(--unaided)
    dest="$OUT/unaided"
  fi
  for seed in 1701 1707 1711; do
    for arm in cog attention; do
      "$PY" train.py --data "$DATA" --out "$dest" --arm "$arm" --seed "$seed" "${opts[@]}"
      "$PY" evaluate.py --data "$DATA" --out "$dest" --arm "$arm" --seed "$seed" "${opts[@]}"
    done
    "$PY" evaluate.py --data "$DATA" --out "$dest" --arm no-memory --seed "$seed" "${opts[@]}"
  done
done
"$PY" evaluate.py --data "$DATA" --out "$OUT" --arm heuristic --seed 1701
"$PY" audit099.py --data "$DATA" --evidence "$OUT" --out "$OUT/audit.json"
