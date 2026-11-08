# LRC-ROUTE-REPAIR-002: portable frozen-model handoff

These are the original repair artifacts, with their original bytes and hashes.
The repaired source is already applied at
[`experiments/lrc_route_repair`](../../experiments/lrc_route_repair).
The patch is provided for portability; **do not apply it again on this branch**.

| File | Bytes | Purpose |
| --- | ---: | --- |
| [LRC_ROUTE_REPAIR_002_Checkpoints_and_Evidence_20261108.zip](LRC_ROUTE_REPAIR_002_Checkpoints_and_Evidence_20261108.zip) | 5,489,762 | Twelve confirmation checkpoints, separately labeled development checkpoints, data, predictions, completed audit and report |
| [LRC_ROUTE_REPAIR_002_Code.patch](LRC_ROUTE_REPAIR_002_Code.patch) | 184,354 | Exact 16-file repaired-source patch |
| [lrc_vb001_matched_comparison.zip](lrc_vb001_matched_comparison.zip) | 7,358,645 | Required `lrc_vb001_comparison` source package and its historical VB-001 comparison |

The original dependency's results are historical, separate from the repaired
cohort. The interrupted seven-fit V4 study remains incomplete and is not promoted
by this handoff. Restoring and loading these files performs no retraining or new
refinement experiment.

## Clone, verify and restore

```bash
git clone --branch research/lrc-route-repair-002-handoff-20261108 --single-branch \
  https://github.com/FPC-effortless/cdl-attention-experiment.git
cd cdl-attention-experiment
python3 artifacts/lrc-route-repair-002/restore.py --verify-only
python3 artifacts/lrc-route-repair-002/restore.py
```

The helper uses only Python's standard library (Python 3.11 or later), checks
all three SHA-256 hashes and ZIP integrity, and extracts both archives into
`artifacts/lrc-route-repair-002/restored/`. It checks for unsafe archive paths and
refuses to overwrite differing existing files. Repeating it reuses matching
files. `--destination PATH` selects another extraction directory. The archives
expand to 187,209,477 bytes (about 187 MB / 179 MiB) in total; the extraction
directory is ignored by Git.

The same hashes are independently checkable with `sha256sum -c SHA256SUMS`
from this directory.

## Restore the recorded CPU environment

The prefit manifest records **CPython 3.12.14**, **PyTorch 2.5.1+cpu**, Linux
x86_64, and one Torch thread. NumPy **2.3.5** was observed in the packaging
environment; the original prefit manifest did not record a NumPy version.
NumPy is needed for the original data-digest/audit code's tensor conversions.
Use CPython 3.12.14 when matching the recorded environment. The following assumes
`python3.12` selects that interpreter:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r artifacts/lrc-route-repair-002/requirements-restore.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/artifacts/lrc-route-repair-002/restored"
```

Use the parent of the extracted `lrc_vb001_comparison` directory in `PYTHONPATH`
if you selected another destination. Commands below run from the repository root.
Matching software versions does not reproduce another machine's latency.

## Check structural behavior and load the frozen repair

```bash
python -m unittest -v \
  experiments.lrc_route_repair.tests_index \
  experiments.lrc_route_repair.tests_model
python - <<'PY'
from pathlib import Path
import torch
from experiments.lrc_route_repair.run import check_frozen, load_model

torch.set_num_threads(1)
output = Path("artifacts/lrc-route-repair-002/restored/LRC_ROUTE_REPAIR_002/confirmation")
manifest = check_frozen(output)
for seed in (901, 902, 903):
    model = load_model(output, "repair", seed)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    print(f"Loaded frozen repair_{seed}.pt: {sum(p.numel() for p in model.parameters())} parameters")
print(f"Verified {len(manifest['source_sha256'])} frozen source hashes")
PY
```

The three reference checkpoints are:

- `restored/LRC_ROUTE_REPAIR_002/confirmation/repair_901.pt`
- `restored/LRC_ROUTE_REPAIR_002/confirmation/repair_902.pt`
- `restored/LRC_ROUTE_REPAIR_002/confirmation/repair_903.pt`

Paths above are relative to this handoff directory. MemNet and negative-control
checkpoints are alongside them. Use the confirmation checkpoints for the
frozen-weight refinement diagnostic. The `development/` checkpoints have a
different role and remain explicitly separated.

For an optional independent integrity/replay audit, write a **new** output file:

```bash
python -m experiments.lrc_route_repair.audit \
  --output artifacts/lrc-route-repair-002/restored/LRC_ROUTE_REPAIR_002/confirmation \
  --source artifacts/lrc-route-repair-002/restored/lrc_vb001_comparison \
  --audit-output artifacts/lrc-route-repair-002/restored/replay_audit_new.json
```

The audit refuses to overwrite an existing output. Keep the archived
`independent_audit.json` unchanged; choose another new output name on later
replays. An audit replay checks existing evidence and does not repeat training
or the full timing benchmark.

## Provenance

The original run records frozen source commit
`6c3976efc2382e28731009f9637ecf31872e5c13`; the completed local review recorded
`3f71af6eb53827eb7eb3c74e10964f70e8e84460`. These are historical local provenance
references, not prerequisites to fetch or promises that those commits exist on
GitHub. This publication branch has its own commit identity. Its frozen source
is checked against the archived prefit manifest by `check_frozen` above.

The code patch exactly represents changes to `experiments/lrc_route_repair`
between local references `6b8ca19` and `3f71af6`. The evidence manifest verifies
61 files; the prefit manifest covers 23 repair/original source files. This
handoff preserves the artifacts and source rather than regenerating results.
