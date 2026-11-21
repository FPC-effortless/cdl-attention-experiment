# Reproduce COG-ATTN-092–095

Requires Python 3.12+ standard library, and the parent stacked research branches (#75–#77) in the same checkout. No network access, external model downloads, or GPU are required for this study.

From repository root:

```bash
cd research/cog-081-086
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
p=Path('extreme/artifacts/puzzle-087-090-cases-evidence.zip')
with ZipFile(p) as z:
    z.extract('results/cases.jsonl',path='extreme')
PY
cd extreme/followup_092_095
export PYTHONHASHSEED=0
python maze_092.py
python sliding_093.py
python sliding_fast_093b.py
python sliding_fresh_093c.py
python sudoku_094.py
python chess_095.py
python verify_stable.py
python -m unittest -v test_092_095.py
```

`verify_stable.py` is based on the committed `STABLE_SHA256.json`. It excludes CPU timings (e.g. `elapsed_ns`, `policy_ms`, `train_ms` and repeated timing traces), but verifies deterministic decisions and work counters. Never update the frozen receipt after observing a mismatch without documenting a new protocol revision, environment and source pin.

The 32 additional 8-puzzle states are **exact-depth certified by evaluator-only full state-space traversal**, which is not imported by inference policies. The test verifies their exact success path lengths. Timing comparisons use three within-host randomized arm orders and median aggregate CPU time across repetitions, not fake synchronized hardware performance. Full per-task traces and original on-host timer records remain in `results`.
