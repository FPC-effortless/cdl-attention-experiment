# VB-001 reproduction

Read [PROTOCOL.md](PROTOCOL.md) for the frozen design and limitations. The final
report is [results/vb-001-2026-11-08.md](../../results/vb-001-2026-11-08.md).

This is a new structured-template task. It does not reproduce the unavailable
versioned LRC experiment or its 100% / 23.94% figures. No LRC win/loss follows.

From the repository root, with Python 3.12:

```sh
python -m pip install -r relational_composition/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r relational_composition/requirements-inference.txt
python -m unittest relational_composition.versioned_baselines.test_study -v
python -m relational_composition.versioned_baselines.study run --out /absolute/new/evidence-directory
python -m relational_composition.versioned_baselines.audit /absolute/new/evidence-directory
```

The output directory must not already exist. All 15 fits precede held-out test
generation. The run saves every checkpoint, optimizer/RNG state, data tensor,
loss trace and prediction. `audit` reloads every final checkpoint, verifies
minibatch/data/source hashes, exactly replays all metrics/predictions and measures
interleaved batch-one CPU latency. It overwrites its audit/timing output files
when rerun; copy the published originals first if comparing hardware timings.

Executed environment: Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5, one CPU
thread. Full fits took about 155 seconds here, excluding setup, evaluation and
audit. Timing is hardware dependent; equality of floating-point replay is scoped
to this environment. The study source SHA-256 is in every checkpoint receipt.

The prospectively frozen local commit is
`f3287dbc5692968aef4463aece041d21b8e074e9`. GitHub protocol commit
`9f50f3ffd6d05565be6ccbd766bc72180a743e80` has the identical tree
`b9794d350c8e835a8543adcb37500d607e0dbea9`; remote publication occurred after fitting.
This is not an independent preregistration or an independent external audit.
