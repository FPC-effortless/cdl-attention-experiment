#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# A different CPython stdlib source is not the same experimental corpus.
python audit_compact.py
for seed in 1701 1707 1711; do
  python experiment.py --seed "$seed" --epochs 8 --out "result-$seed.json" --checkpoint "model-$seed.pt"
  python feedback.py --seed "$seed" --out "feedback-$seed.json" --weights-dir .
done
python audit.py --replay
python -m pytest -q test_experiment.py test_feedback.py
