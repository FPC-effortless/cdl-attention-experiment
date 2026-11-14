# Reproduction

Restore this source at its published commit plus the private evidence archive alongside the parent LIFECYCLE-LEARNING-001 evidence. The parent source is included through the base branch. Use the recorded Python3.12/Torch2.5.1 CPU and ecosystem source trees; `run.check()` validates exact source/version identities, not a full operating-system attestation. The recorded torch path is environment-specific; do not silently relax it. A portable rerun requires a separately documented environment admission.

For the completed run, verify TrainLab's receipt with `verify_run(path)`, inspect checkpoint before/after hashes, and recompute `summarize(json.loads(predictions_path.read_text()))`. The source intentionally retains the executed reporting size-limit failure for audit; use `finalize.py` only for reporting recovery of that existing run. Do not invoke the one-shot `run` command on an evidence directory containing a completed run. No model fitting is needed for verification.

The three tests can be run with `python -m unittest test_probe -v` from this directory. Independent review scripts and receipts are included with the evidence package. The pre-inference freeze covers run.py, protocol, tests, dataset and checkpoints; finalize.py and RESULTS.md are explicitly post-execution reporting artifacts.
