# LRC-TS-001

New, parameter-matched Tiny Shakespeare training and benchmarking cohort for
every recovered general LRC language architecture and applicable prose recipe.
Read [PROTOCOL.md](PROTOCOL.md) and [VERSION_INVENTORY.md](VERSION_INVENTORY.md)
before interpreting a rank. Historical Sherlock Holmes scores are not mixed in.

## Reproduce

Python3.12, PyTorch2.5.1 CPU, NumPy. Use a virtual environment and install the CPU
wheel from the official PyTorch index. Source and the1,115,394-byte public-domain
corpus are included; the runner checks its canonical SHA-256.

From this directory:

```bash
python test_integrity.py
python test_protocol.py
python run.py prepare
python launch.py --workers 6
```

The frozen run comprises42 backbone fits (14 arms ×3 seeds) plus9 final utility
head fits. Training budgets are2,048 updates, batch4, context128, and1,048,576
target-byte exposures per backbone. The600-update gate stage is additional work.
Workers each use one CPU compute thread. All fitting finishes before test scores
are opened; inference and training-speed measurements run after workers exit.

`artifacts/checkpoints` contains model, optimizer and RNG states for backbone
fits, and gate heads with parent-checkpoint hashes for post-training. Checkpoints
are excluded from Git and supplied as a separate archive. Receipts, compressed
per-update logs, block-level test scores, timing repeats, samples and summaries
are retained under `artifacts`. `fits_closed.json` is the checkpoint freeze record.

For individual stages:

```bash
python run.py train --arm v4 --seed 2201
python run.py gates --seed 2201
python run.py close
python run.py evaluate --arm v4 --seed 2201
python run.py profile
python run.py summarize
```

Existing completed fits are reused only after identity checks. A partial gate
cohort or changed source fails explicitly. Evaluation requires the complete
cohort's freeze record. The final checkpoint is always used; no test-based model
selection or exclusion of weak seeds is allowed.

## Interpretation

Primary lower-is-better measures: bits/byte, next-byte NLL and byte perplexity.
Next-byte accuracy and throughput are higher-is-better. Memory-off minus normal
BPB is positive when the trained checkpoint benefits from its memory. It is
distinct from comparing two independently trained architectures.

The models allocate roughly43K parameters, with exact sizes disclosed. This
is approximately one corpus-equivalent of sampled training positions, not a
complete epoch and not evidence of convergence. Historical architecture shapes
are normalized to this budget. The two-page-head configurations differ in width,
and the larger head's capacity is not exercised fully at the evaluation lengths.
The raw-byte scores cannot be compared directly with historical BPE-token PPL.
Profiler FLOPs are partial operator counts; timings include indexing and state
maintenance. CPU measurements do not establish GPU speed or frontier scaling.
