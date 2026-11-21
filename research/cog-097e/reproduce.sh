#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q test_experiment.py
for seed in 1701 1707 1711; do
  python experiment.py --seed "$seed" --epochs 3 --out "result-$seed.json" --weights .
done
python audit.py --folder .
python summarize.py
