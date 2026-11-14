# Inspect or reproduce this bounded research diagnostic

The companion evidence package contains the frozen Forge release, all six checkpoints and optimizer states,512-step logs, TrainLab receipts, raw predictions, complete summaries, source/runtime hashes and independent review. Source is in this directory; no source file from the native CSM was altered.

Use Python3.12,Torch2.5.1+cpu and the existing Forge/TrainLab/Veritas/Research Lab packages. The recorded ecosystem runtime is the local Veritas lifecycle-observability source from PR514; exact source-tree hashes are in freeze.json. GPU execution was not used. Dependency versions and executable paths in old receipts describe that execution, not a portable command to edit in place.

For inspection, restore the evidence files beside the matching source and compare their hashes. `analyze.py` recomputes descriptive summaries from saved predictions. Independent reviewer replay is supplied in the evidence package. Do not treat supplied-control or deterministic-proposal tests as learned results.

For a NEW reproduction, copy only PROTOCOL.md,data.py,models.py,native_csm.py,experiment.py,test_research.py into a fresh directory with the four ecosystem packages installed. Run the source checks, then `python experiment.py prepare`; retain the printed dispatch hash. Run `python experiment.py fit <dispatch-hash>`, then `python experiment.py evaluate` only after all fits close. The experiment is small but still six explicit CPU training jobs; the commands are trusted local code. A different runtime/executable creates a new preparation and new receipts. Never edit old pins or overwrite attempts to manufacture a replay.

The source tests independently check all generated semantics before fitting. Forge's third bucket is not exported for neural scoring. The original model comparison remains a known-template supervised feasibility study, not an admitted architecture benchmark.
