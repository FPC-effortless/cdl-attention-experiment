# Independent decoder and tensor persistence validation

The repaired engineering boundary passed the final adversarial suite and all fresh-process continuation/restore checks. This does **not** establish learned semantic memory. The model weights are seeded, untrained reference fixtures; no fitting, language benchmark, or semantic answer scoring occurred.

## Failures found and repaired

The baseline suite ran 19 independent tests: 13 passed and 6 failed. Those six failures covered four concrete issues:

| Failure | Observed baseline behavior | Repair |
|---|---|---|
| Convolution execution identity, live and reopen cases | Changing dilation and padding retained the same parameter and output shapes, so the old fingerprint accepted different computation | Bind execution attributes of supported layers; reject model identity changes |
| Global forward and pre-forward hooks | Hooks could change output logits or substitute token inputs without changing the old fingerprint | Reject global inference hooks, per-instance execution replacement, class-level replacement after trusted import, and compiled call replacement |
| First checkpoint versus source erasure | A first write could commit after source deletion because no older receipt existed to invalidate it | Snapshot scope erasure generation before reads/computation; compare atomically during commit |
| Receipt revision corruption | Changing a cached revision from 1 to 101 still returned a successful old retry | Store seals now bind metadata to payload/result hashes and verify them before use |

All original failing cases now pass. Five additional cases cover class-level LocalOnly and Linear replacements, compiled calls, sparse token input, and the explicit concurrent-erasure retry boundary. Final decoder/codec suite: **24 passed, 0 failed, 0 errors**. The wider storage/system suite is reported separately by the parent validation report.

The execution checks assume a trusted Python interpreter. They do not authenticate arbitrary code installed before import, code that changes the validator itself, or an attacker who can rewrite database content and its checksums.

## Actual reference models across fresh processes

Each model uses vocabulary 65, width 128, six layers, seed 20261123, and smear gain 0.7. The same 257-token synthetic sequence is split into lengths **1, 2, 7, 31, 3, 64, 5, 97, 47**. Every chunk starts a new Python interpreter, recreates identical model weights, opens the database, restores state, and commits its next result.

Persisted logits and the complete restored state are compared byte for byte with an in-memory control using identical chunk boundaries. A separate uninterrupted 257-token computation is compared with `rtol=1e-5, atol=1e-6`. Full-sequence equality is tolerance based because matrix shapes differ across chunks.

| Mixer | Parameters | Fresh-process chunks exact | Restored continuations exact | Max absolute logit difference vs uninterrupted | Changed argmax positions |
|---|---:|---:|---:|---:|---:|
| LocalOnly | 919,846 | 9/9 | 4/4 | 8.34465027e-07 | 0/257 |
| PacketMixer | 1,074,358 | 9/9 | 4/4 | 1.43051147e-06 | 0/257 |
| PacketAndTypedProbe | 1,218,832 | 9/9 | 4/4 | 9.53674316e-07 | 0/257 |

This is **27/27 exact chunk/state comparisons**, **12/12 exact continuations from restored backups**, and **3/3 exact old-retry replays without changing the latest state**: 42 separate worker invocations. All three complete sequences remained within the declared numerical tolerance; none changed the selected maximum-logit token versus uninterrupted inference.

Backup is taken after the fifth chunk, at 44 consumed tokens, then restored to a new database path. Four further processes consume the remaining 213 tokens on each restored branch. The following sizes are local SQLite fixture snapshots, not lifetime storage estimates:

| Mixer | Backup database bytes | Restored initial tensor state exact |
|---|---:|---|
| LocalOnly | 212,992 | True |
| PacketMixer | 290,816 | True |
| PacketAndTypedProbe | 294,912 | True |

Runtime: CPU, one Torch thread, Torch 2.10.0+cpu, SQLite 3.53.1. The final exercise took 118.57 seconds including 42 interpreter launches and checks. This is test duration, not a latency or throughput comparison.

## What the state actually influences

A separate 31-token suffix is evaluated with three conditions: its actual 257-token prefix state, empty state, and state from a different 257-token synthetic prefix. Each model's weights and suffix tokens stay fixed.

| Mixer | Max logit change with empty state | Changed argmax positions | Max logit change with wrong prefix state | Changed argmax positions |
|---|---:|---:|---:|---:|
| LocalOnly | 0.992799 | 3/31 | 1.439772 | 4/31 |
| PacketMixer | 0.619979 | 3/31 | 0.709705 | 3/31 |
| PacketAndTypedProbe | 1.755183 | 21/31 | 1.628611 | 19/31 |

These interventions show that saved state affects computation. There are no semantic ground-truth labels here, so these changes do not show whether the remembered content is correct, relevant, useful, or robust to unfamiliar wording. They cannot establish a repair to PULSE writing, temporal action binding, item copying, or full-history accuracy.

## The durable-fact integration boundary

The test also creates two genuinely different accepted ledger facts: `project.launch_colour = red` and `project.launch_colour = blue`. Both stores contain their corresponding source observation and accepted proposal. With the same model and supplied token sequence, **all three decoders produce bit-identical logits; maximum difference is 0**.

This is an explicit capability boundary: `PersistentDecoder` receives the tokens supplied by its caller. It does not query, select, tokenize, or inject the durable semantic ledger. A host using `MemorySession` must wire retrieved records into its model callback. This experiment does not demonstrate automatic factual memory across independent chats, and no artificial injection was added to manufacture a sensitivity pass.

## Upgrade and erasure boundaries

The upgrade probe builds a real v1 database with the published `56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f` code in an isolated process. It stores a source observation, an accepted fact, and actual decoder tensor state, then opens that database with the repaired implementation.

- SQL schema migrates from v1 to v2; the observation and accepted fact remain available.
- The old decoder checkpoint remains stored and its tensors still satisfy shape/chronology validation.
- Attaching the strengthened decoder to that old session is rejected with `IntegrityError` because the execution-identity format changed from manifest v1 to v2.
- A fresh transient stream succeeds under manifest v2 and begins its token count at the new input length.

Thus model/runtime upgrades do **not** transparently preserve native transient continuation. Durable records and model-specific inference state have different compatibility rules. No automatic bypass or unverified conversion was introduced.

New inference linearizes at its atomic commit. Scope erasure that wins before commit prevents a stale new state from being published, including a first-ever checkpoint. An exact retry linearizes at its receipt read: an overlapping deletion cannot revoke bytes already read into that caller. The dedicated test demonstrates that the overlapping read can return its previously read result, while the next read is expired. This is not a claim of post-read revocation.

## Evidence and reproduction

Accepted final evidence is `final.json`, `system_results.json`, and `legacy_upgrade.json`. `baseline.json`/`.txt` preserve the original six failures. `mixed_revision_run.json` is explicitly invalid: production source changed during that development run. `pre_manifest_v2_system_results.json` is a successful earlier frozen snapshot, superseded by the final version-matched run.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python research/nanocog-001/memory-validation-002/decoder/run_checks.py --label final
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python research/nanocog-001/memory-validation-002/decoder/run_frozen_system.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python research/nanocog-001/memory-validation-002/decoder/legacy_upgrade_probe.py
python research/nanocog-001/memory-validation-002/decoder/write_report.py
```

The upgrade probe requires the published baseline commit to exist in the local Git object store. The final multi-process runner freezes source before starting workers, avoiding mixed implementations during concurrent development.

Final source SHA-256:

| File | SHA-256 |
|---|---|
| `nanocog/memory/decoder.py` | `65e5afb321dbab4014c8aa11d860b272f5003fb5758064dfe22e76a27c771b0a` |
| `nanocog/memory/store.py` | `3bf7561d194e19cc32a6ab4377ca4656aea86800ce6d026fd0a291ab18ec47c1` |
| `nanocog/memory/codec.py` | `15761577fef3dc0769e817b7da7a5a7d29a0d89b4e656f41e553de86204f112f` |

No trained checkpoint qualification, physical power-loss test, deployment test, long-duration operating history, or claim of lifetime reliability is supplied by this decoder report.
