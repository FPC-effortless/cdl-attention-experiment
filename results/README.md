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

## LRC route repair

- [LRC-ROUTE-REPAIR-002 — completed repair report](lrc-route-repair-002-2026-11-08.md): frozen sentence-level repair, three confirmation seeds, held-out predictions, and independent audit. The repair passes the controlled capability gate; MemNet remains faster.
- [Download and restore the repair](../artifacts/lrc-route-repair-002/README.md): exact evidence/checkpoint ZIP, implementation patch, original comparison dependency, checksums, and commands for frozen-weight use without retraining. The repaired implementation is included under `experiments/lrc_route_repair/`.
