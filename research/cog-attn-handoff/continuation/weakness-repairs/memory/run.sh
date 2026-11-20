#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 181 193 211; do
    TRAINLAB_SEED="$seed" python memory/confirmed_trial.py candidate
    TRAINLAB_SEED="$seed" python memory/confirmed_trial.py control
done > memory.jsonl
TRAINLAB_SEED=181 TRAINLAB_OUTPUT_DIR="$PWD/memory" python memory/drift_profile.py candidate
