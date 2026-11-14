# Reproduction and audit

Restore the exact published source commit, then overlay private evidence research/ entries only where files are absent or identical. The parent implementation is present through branch history; initial checkpoints and parent closure/receipt files are in the evidence package. Historical parent datasets are not needed. Recorded Python/Torch/package versions and ecosystem source hashes must match; an alternate environment requires separate requalification.

The completed study uses `experiment.py prepare`, then the exact pinned `fit` dispatch, then `evaluate` only after all12 fits close. Do not reissue one-shot preparation/training/evaluation in a completed evidence directory. `repair_analysis.py` recomputes summaries from saved predictions/state vectors without fitting; `experiment.py register` verifies and imports the completed evidence into the existing research graph.

Each fit has a TrainLab receipt,256 update records, final optimizer/model/RNG state and initial/schedule hashes. Each seed shares one data/query schedule across the two architectures and two continuation objectives. Each architecture/seed control and repair starts at exactly the same existing checkpoint and a fresh AdamW state.

The validation file contains80 paired groups, two worlds and four queries each:640 queries per model/condition. Eighteen evaluated checkpoints (six starting/twelve continued) × two conditions yields23,040 validation predictions. The secondary288-case diagnostic yields5,184 additional predictions. The third Forge partition receives no model scoring. Integrity/semantic preparation may inspect generated data; this is not model validation.

The normal-validation root vectors are aligned by trace_index and capture token17 with no change to model execution. Scale diagnostics cannot prove useful invariance. Independent replay is audit evidence of the same models/cases, not another scientific replicate.
