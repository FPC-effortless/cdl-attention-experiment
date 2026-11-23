# MEMORY-VALIDATION-002: adversarial validation and measured inference cost

## Scope and evidence order

The user requested tests of the newly built memory infrastructure and model path.
The starting implementation is commit
`56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f` on
`feat/durable-memory-001` (draft PR 94). Its existing CPU suite reported 88 passes
and one optional archived-source skip; green CI is not a substitute for the
independent failure probes in this study.

This protocol records the consolidated scope after the independent storage and
decoder auditors had reported their initial defects. It is **not a retrospective
preregistration of those probes**. The scale workload has its own protocol and
pinned baseline package. The inference cost plan below was written before its
first timing run. Preserve baseline failures rather than rewriting their results
after repair.

No new training, paid compute, deployment or merge is part of this study. Use
synthetic observations, identities, inputs and deterministic untrained reference
weights. The private historical PULSE source may be used only for the existing
hash-pinned source-integration test; do not publish private archives or weights.

## Acceptance criteria

1. **Durability and scope:** successful acknowledged operations survive a new OS
   process. Crashes before a transaction commits leave no partial operation.
   Conflicting processes cannot overwrite a newer revision. Another scope must
   not affect a fixed scope's results or ordering.
2. **Useful bounded recall:** direct accepted key/value matches must not be
   displaced by unrelated facts sharing a source. The returned context remains
   within its declared record and serialized-character budgets. These are
   lexical fixtures, not a claim of paraphrase or semantic retrieval competence.
3. **Erasure:** recorded dependent messages and facts are removed; old requests
   do not resurrect them. Decoder work based on a pre-erasure scope snapshot
   cannot commit, including the first checkpoint. Previously exported backups
   retain their separately documented erasure limitation.
4. **Decoder continuation:** compare resumed outputs to the same model and input
   in uninterrupted inference, and compare them to reset/wrong-state controls.
   Require the declared floating tolerance. Numerical state influence does not
   imply learned correct answers or indefinite retention of arbitrary facts.
5. **Identity and corruption:** reject incompatible models/tokenizers, modified
   execution properties, unsupported hooks, corrupted state and inconsistent
   checkpoint metadata. This is accidental-corruption detection in a trusted
   host process, not cryptographic authentication against a database writer.
6. **Upgrade and recovery:** schema changes require tested migration of existing
   stores; backups and restored stores preserve the relevant invariants.

## Work packages

| Package | Owner | Artifacts |
|---|---|---|
| Multiprocess storage, source/deletion races, recall interference | storage auditor | `storage/` |
| Actual Decoder, packet/typed states, restarts, identity, receipts | decoder auditor | `decoder/` |
| 1k/10k/50k archive, fresh process, backup, CLI, temporal reference | scale auditor | `scale/` |
| Inference cost, whole-suite gate, report and publication | parent | `inference/`, `RESULT.md` |

Fixes are developed against disclosed synthetic failures, so the post-fix suite
is regression verification, not fresh held-out qualification of a learned model.

## Inference cost plan (before measurements)

Use CPU float32, one PyTorch worker thread, seed 91023, vocabulary 256, width
160, four layers, batch one. Test LocalOnly (1,005,474 parameters), PacketMixer
(1,134,210), and PacketAndTypedProbe (1,274,942). Geometry is fixed; parameter
counts differ. These models are untrained engineering fixtures. Give every path
within a model exactly the same weights and token inputs.

Use a temporary directory under the workspace overlay filesystem, not the
default `/tmp` tmpfs. Record the filesystem returned by `df -PT`. This measures
the available host filesystem and process-failure behavior; it cannot prove
physical-drive or power-loss durability. The scale auditor identified the tmpfs
distinction before this inference timing run; its earlier tmpfs numbers remain
labelled separately from its subsequent overlay measurements.

For history lengths 128 and 1,024, and next-chunk lengths 1 and 16, compare:

- recomputing the full prefix plus chunk;
- direct in-RAM incremental continuation from the prefix state;
- a new committed PersistentDecoder continuation from the persisted prefix;
- an exact request retry returning the stored receipt.

Use one warm-up and five measured trials, fresh independent persisted sessions
for new commits, and report median, nearest-rank p95, all raw samples, source
hashes, software and host details. Database reopen/model initialization are
outside these warm measurements. Separately measure a fresh subprocess that
initializes the deterministic model, opens the DB and continues the same stream;
retain subprocess wall time including Python/PyTorch startup. Prefix creation
cost is outside next-chunk timing and must be labelled. Output serialization,
fingerprint checks, state load/save and SQLite commit are inside persistent timing.

Require all paths' tail logits to agree with uninterrupted inference at
`rtol=1e-5, atol=1e-6`; record maximum absolute error. Also report raw and encoded
state size, encoded output size, a standalone identity-check timing, and a
standalone state codec round trip. Run primary timings after other test workers
finish to reduce contention. This is one-host descriptive timing with no trained
accuracy comparison, throughput or hardware-lifetime claim. Persistence may be
slower than a short full-prefix recomputation; retain unfavorable results.

## Completion gate

An independent review of the inference harness found no timing-boundary or
shared-state blocker, but requested explicit checks of cold-process revision and
token count, and inclusion of the harness itself in the source-change guard.
The initial timing run is retained as `inference/initial-results.json`; the
primary run repeats the identical workloads with those verification-only
changes. Model weights, inputs, measured trials and timing boundaries are
unchanged. This amendment was recorded before that primary repeat.

Keep each independently reproduced baseline failure with its exact script and
baseline source identity. After fixes, rerun the affected probes and the scale
workload. Then run the complete package suite once, enabling the optional PULSE
source integration only after matching both required source hashes. Report any
remaining failures, unsupported paths and untested claims explicitly. Publish
code and synthetic evidence on the existing review branch; do not merge it.
