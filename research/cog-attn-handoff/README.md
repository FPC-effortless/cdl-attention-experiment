# COG-ATTN research handoff (2026-11-20)

**Continuation through the 047–050 qualification attempt and its follow-ups is available.** The [continuation index](continuation/README.md) links browsable source and original evidence bundles in the [binary evidence branch](https://github.com/FPC-effortless/cdl-attention-experiment/tree/research/cog-attn-continuation-evidence-20261120). The partial status below applies to the original 001–046 and WR-001–003 handoff.

**Evidence publication is partial.** This branch contains the ten original reports, eight protocol documents, and the original WR-001–003 Python source files. These 21 original files were matched byte-for-byte against their original Git blob hashes. The ten original evidence archives contain 241 files, including 69 checkpoints. Their archive checksums and all nine available internal manifests were verified locally, but the binary archives and remaining files have NOT yet been pushed. Do not treat this branch as a self-contained reproduction until the evidence import is complete.

## Experiment index

| Cycle | Reproduction notes |
| --- | --- |
| 001 | [Pilot](001/README.md) |
| 002–010 | [Attention and memory experiments](002-010/README.md) |
| 011–013 | [Intermediate experiments](011-013/README.md) |
| 014–020 | [Report](014-020/README.md), [protocol](014-020/protocol.md) |
| 021–030 | [Report](021-030/README.md), [protocol](021-030/protocol.md) |
| 031–040 | [Report](031-040/README.md), [protocol](031-040/protocol.md) |
| 041–046 | [Report](041-046/README.md), [protocol](041-046/protocol.md) |
| Epistemic lifecycle V1 | [Report](lifecycle/README.md), [protocol](lifecycle/protocol.md) |
| Epistemic lifecycle V2 | [Report](lifecycle-v2/README.md), [protocol](lifecycle-v2/protocol.md), [follow-up](lifecycle-v2/followup_protocol.md) |
| WR-001–003 | [Report](weakness-repair/README.md), [protocol](weakness-repair/protocol.md), [run.py](weakness-repair/run.py), [followup.py](weakness-repair/followup.py), [audit.py](weakness-repair/audit.py) |

## Import and reproduction

The original archive byte digests are in [ARCHIVE_SHA256SUMS](ARCHIVE_SHA256SUMS). The author has a separate complete handoff package containing these ten ZIP files and an import script. Obtain that package from the author; it is not hosted here yet. The import script copies the unchanged archives into evidence/archives and their members into evidence/unpacked, checks the hashes, and publishes only on this research branch. After publication verify all ten digests and 241 extracted files. Use the original experiment READMEs, protocols and native environment requirements; reproduce in a separate working copy so original results and checkpoints remain immutable. External datasets/upstream revisions may require network access and their original license conditions. Treat checkpoints as executable serialization unless inspected and safely loaded.

For the latest synthetic weakness-repair pilot, run its three Python scripts in the documented order in a clean environment; retain any newly generated checkpoints/results separately. The earlier report's audit checked saved checkpoint evaluation, not independent retraining. A local checksum or replay is not an independent replication.

## Continuation priorities

- WR-004: raw-text semantic grounding without supplied entity spans or roles.
- WR-005: bounded candidate proposals and consequence credit assignment, with candidate coverage and exploration.
- WR-006: bitemporal episodic indexing and worst-case query/correction cost.
- WR-007–009: learned escalation, original trained-module integration, latency and provenance overhead.
- COG-ATTN-048 → 047 → 049 → 050: actual trained-module integration; grounded multi-turn language; competent causal baselines; authentic end-to-end continual evaluation. These remain outstanding, as does WR-010. Preserve negative findings and predeclare gates, separate coverage from precision, and report cold-start, verification, repair, memory and steady-state latency. Inference is weighted 12x relative to training.

Do not claim a general-purpose Transformer replacement: the present evidence is bounded, includes synthetic tasks and specialized controls, and has not jointly qualified authentic task accuracy and full end-to-end inference performance. Record source revisions, seeds, configs, dataset snapshots, commands, checkpoints, raw per-seed results, failures, and independent audit evidence for future experiments.
