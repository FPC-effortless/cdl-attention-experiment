"""Render the decoder report only after every final evidence gate passed."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
checks = json.loads((HERE / "final.json").read_text())
system = json.loads((HERE / "system_results.json").read_text())
upgrade = json.loads((HERE / "legacy_upgrade.json").read_text())
assert checks["success"] and system["passed"] and upgrade["passed"]
assert system["accepted_as_validation"] and system["frozen_source_snapshot"]
for source in ("nanocog/memory/decoder.py", "nanocog/memory/store.py", "nanocog/memory/codec.py"):
    assert (checks["source_sha256_after"][source] == system["source_sha256_after"][source]
            == upgrade["source_sha256_after"][source])

rows = []
influence = []
backup = []
for model in system["models"]:
    name = {"local": "LocalOnly", "packet": "PacketMixer", "probe": "PacketAndTypedProbe"}[model["kind"]]
    chunks = sum(row["passed"] for row in model["fresh_process_chunks"])
    restores = sum(row["passed"] for row in model["backup_restore"]["continuation"])
    full = model["uninterrupted_comparison"]
    rows.append(f"| {name} | {model['parameters']:,} | {chunks}/9 | {restores}/4 | {full['max_abs_difference']:.9g} | {full['argmax_changed_positions']}/257 |")
    reset = model["numerical_state_influence"]["correct_vs_reset"]
    wrong = model["numerical_state_influence"]["correct_vs_wrong_history"]
    influence.append(f"| {name} | {reset['max_abs_difference']:.6f} | {reset['argmax_changed_positions']}/31 | {wrong['max_abs_difference']:.6f} | {wrong['argmax_changed_positions']}/31 |")
    backup.append(f"| {name} | {model['backup_restore']['backup_bytes']:,} | {model['backup_restore']['restored_state_bit_exact']} |")

report = f"""# Independent decoder and tensor persistence validation

The repaired engineering boundary passed the final adversarial suite and all fresh-process continuation/restore checks. This does **not** establish learned semantic memory. The model weights are seeded, untrained reference fixtures; no fitting, language benchmark, or semantic answer scoring occurred.

## Failures found and repaired

The baseline suite ran 19 independent tests: 13 passed and 6 failed. Those six failures covered four concrete issues:

| Failure | Observed baseline behavior | Repair |
|---|---|---|
| Convolution execution identity, live and reopen cases | Changing dilation and padding retained the same parameter and output shapes, so the old fingerprint accepted different computation | Bind execution attributes of supported layers; reject model identity changes |
| Global forward and pre-forward hooks | Hooks could change output logits or substitute token inputs without changing the old fingerprint | Reject global inference hooks, per-instance execution replacement, class-level replacement after trusted import, and compiled call replacement |
| First checkpoint versus source erasure | A first write could commit after source deletion because no older receipt existed to invalidate it | Snapshot scope erasure generation before reads/computation; compare atomically during commit |
| Receipt revision corruption | Changing a cached revision from 1 to 101 still returned a successful old retry | Store seals now bind metadata to payload/result hashes and verify them before use |

All original failing cases now pass. Five additional cases cover class-level LocalOnly and Linear replacements, compiled calls, sparse token input, and the explicit concurrent-erasure retry boundary. Final decoder/codec suite: **{checks['tests_run']} passed, {checks['failures']} failed, {checks['errors']} errors**. The wider storage/system suite is reported separately by the parent validation report.

The execution checks assume a trusted Python interpreter. They do not authenticate arbitrary code installed before import, code that changes the validator itself, or an attacker who can rewrite database content and its checksums.

## Actual reference models across fresh processes

Each model uses vocabulary 65, width 128, six layers, seed 20261123, and smear gain 0.7. The same 257-token synthetic sequence is split into lengths **1, 2, 7, 31, 3, 64, 5, 97, 47**. Every chunk starts a new Python interpreter, recreates identical model weights, opens the database, restores state, and commits its next result.

Persisted logits and the complete restored state are compared byte for byte with an in-memory control using identical chunk boundaries. A separate uninterrupted 257-token computation is compared with `rtol=1e-5, atol=1e-6`. Full-sequence equality is tolerance based because matrix shapes differ across chunks.

| Mixer | Parameters | Fresh-process chunks exact | Restored continuations exact | Max absolute logit difference vs uninterrupted | Changed argmax positions |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

This is **27/27 exact chunk/state comparisons**, **12/12 exact continuations from restored backups**, and **3/3 exact old-retry replays without changing the latest state**: 42 separate worker invocations. All three complete sequences remained within the declared numerical tolerance; none changed the selected maximum-logit token versus uninterrupted inference.

Backup is taken after the fifth chunk, at 44 consumed tokens, then restored to a new database path. Four further processes consume the remaining 213 tokens on each restored branch. The following sizes are local SQLite fixture snapshots, not lifetime storage estimates:

| Mixer | Backup database bytes | Restored initial tensor state exact |
|---|---:|---|
{chr(10).join(backup)}

Runtime: CPU, one Torch thread, Torch {system['runtime']['torch']}, SQLite {system['runtime']['sqlite']}. The final exercise took {system['elapsed_seconds']:.2f} seconds including 42 interpreter launches and checks. This is test duration, not a latency or throughput comparison.

## What the state actually influences

A separate 31-token suffix is evaluated with three conditions: its actual 257-token prefix state, empty state, and state from a different 257-token synthetic prefix. Each model's weights and suffix tokens stay fixed.

| Mixer | Max logit change with empty state | Changed argmax positions | Max logit change with wrong prefix state | Changed argmax positions |
|---|---:|---:|---:|---:|
{chr(10).join(influence)}

These interventions show that saved state affects computation. There are no semantic ground-truth labels here, so these changes do not show whether the remembered content is correct, relevant, useful, or robust to unfamiliar wording. They cannot establish a repair to PULSE writing, temporal action binding, item copying, or full-history accuracy.

## The durable-fact integration boundary

The test also creates two genuinely different accepted ledger facts: `project.launch_colour = red` and `project.launch_colour = blue`. Both stores contain their corresponding source observation and accepted proposal. With the same model and supplied token sequence, **all three decoders produce bit-identical logits; maximum difference is 0**.

This is an explicit capability boundary: `PersistentDecoder` receives the tokens supplied by its caller. It does not query, select, tokenize, or inject the durable semantic ledger. A host using `MemorySession` must wire retrieved records into its model callback. This experiment does not demonstrate automatic factual memory across independent chats, and no artificial injection was added to manufacture a sensitivity pass.

## Upgrade and erasure boundaries

The upgrade probe builds a real v1 database with the published `{upgrade['baseline_commit']}` code in an isolated process. It stores a source observation, an accepted fact, and actual decoder tensor state, then opens that database with the repaired implementation.

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
| `nanocog/memory/decoder.py` | `{checks['source_sha256_after']['nanocog/memory/decoder.py']}` |
| `nanocog/memory/store.py` | `{checks['source_sha256_after']['nanocog/memory/store.py']}` |
| `nanocog/memory/codec.py` | `{checks['source_sha256_after']['nanocog/memory/codec.py']}` |

No trained checkpoint qualification, physical power-loss test, deployment test, long-duration operating history, or claim of lifetime reliability is supplied by this decoder report.
"""
(HERE / "REPORT.md").write_text(report)
print(str(HERE / "REPORT.md"))
