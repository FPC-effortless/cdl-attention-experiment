# Results

Store exported benchmark results here after each run.

For every reported result, record at minimum:

- model/revision;
- benchmark seed;
- number of cases;
- candidate memories per case;
- whether native attention was enabled;
- device/hardware;
- aggregate retrieval metrics;
- downstream selected-answer NLL;
- any prompt/template changes made since the previous run.

Do not overwrite earlier runs when changing the benchmark. Preserve them so tuning remains auditable.

## LRC language benchmark

- [LRC-TS-001 — all recovered LRC language variants on Tiny Shakespeare](lrc-ts-001-2026-11-08.md): 42 backbone fits plus nine utility heads; three paired seeds; complete held-out evaluation, CPU efficiency, memory ablations, independent audit, and separate NumPy runtime supplement. Completed 8 November 2026; publication pending approval.
