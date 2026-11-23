# Native persistence serving-cost protocol

Baseline: published `ffb842adc176496d8838b774d0fe412aec5d1d20`.
The original package is frozen outside the working checkout before edits.

## Question and constraints

Can redundant tensor scanning/copying and Python metadata traversal be removed
from the persistent reference decoder while preserving every existing identity,
state, receipt, erasure, resource and durability check? No trained model accuracy
claim, fit, GPU experiment or paid compute is part of this work.

The candidate retains full parameter and buffer content hashing before and after
each new inference and before replay. It must still detect ordinary and `.data`
mutations, including changes to nonfinite values, and cannot substitute PyTorch
version counters for those checks. Metadata/source/method checks remain live.
The manifest format and byte values must match the baseline for unchanged models.

Initial profiling of 30 identities on the 1,274,942-parameter typed fixture found
about 23.6 ms per identity under cProfile. Full hash updates and finite scans are
the largest costs, followed by exception-heavy absent-attribute lookups. This
profile selects engineering targets; it is not a final speed measurement.

## Planned candidate

- Feed canonical contiguous CPU byte views to hashlib instead of constructing
  copied `header + tensor_bytes` buffers. Hash headers and data as consecutive
  updates, preserving the exact SHA-256 input stream.
- Validate model finiteness at attachment. A later full content hash must match
  that finite identity; mismatching bytes fail before use or commit. This avoids
  rescanning all finite tensors independently on each successful check.
- Reuse one live named-module traversal per check. Determine attribute presence
  from instance/class dictionaries, then read present attributes normally, so
  class descriptors and live geometry changes are still observed.
- Keep receipt decoding/validation, erasure generation, expected revisions,
  state validation and synchronous commit semantics unchanged.

Any further change needs a recorded rationale and a specific correctness check.

## Correctness gates

Run the existing decoder/codec and adversarial decoder tests, plus regressions
for `.data` mutation (which may not increment tensor version), nonfinite updates,
unchanged finite byte identity, noncontiguous state buffers, execution attributes
introduced or replaced through descriptors, and cached old-output retries.

Compare baseline and candidate manifests, output tensors and complete checkpoint
payloads for the same model/input. Exercise original v2 checkpoints with the
candidate. A successful optimization must not require a new native stream.

## Cost boundaries

Use CPU float32, one Torch thread, seed 91023, vocabulary 256, width 160, four
layers and batch one for LocalOnly, PacketMixer and PacketAndTypedProbe.
Parameter counts are 1,005,474 / 1,134,210 / 1,274,942. Weights are seeded and
untrained; this measures engineering overhead only.

Test prior history lengths 128 and 1,024, and new chunks of 1 and 16 tokens.
For each variant/case, use one warm-up and five measured trials. Measure full
persistent calls and exact receipt retries, with initialization and prefix
construction excluded from warm timings. Include full fingerprint checks,
state load/validation, model inference, serialization, SQLite commit and output
reconstruction. Record all samples, median and sample-maximum p95. Measure
standalone model identity checks separately; those do not sum to total cost.

The primary optimization comparison uses the same frozen baseline store and
codec/data format, on the workspace overlay filesystem, with implementation
variants executed in separate processes. Alternate execution order across
models. Record source/harness hashes, actual filesystem, versions and complete
timing boundaries. Do not compare old tmpfs measurements to new overlay results.
The integrated optimized-store result is a separate final check, not a substitute
for the isolated decoder comparison.

No output/commit is skipped or delayed in the fast path. No apparent speedup
qualifies if numerical output, complete persisted state, identity mutation
detection, retry behavior or the prior adversarial gates regress.

## Prospective confirmation amendment

After retaining the complete first pass, the parent authorized exactly one
quiet confirmation pass because unchanged RAM controls and persistent timings
varied substantially during overlapping work. Other agents will pause heavy
compute during this pass. The original harness is retained as
`initial_compare.py`; all original results remain in this directory.

Before this confirmation is run, the order is fixed to LocalOnly candidate then
baseline, PacketMixer baseline then candidate, and PacketAndTypedProbe candidate
then baseline. The same seeds, model configurations, histories, chunk lengths,
one warm-up and five measured samples apply. Results will be written only to
`confirmation/`, and both passes will be reported regardless of outcome.

The confirmation also adds exactly one cold-process observation per model and
variant, in the same candidate/baseline order as its warm comparison. The
controller copies the baseline 1,024-token checkpoint before timing begins.
The subprocess wall measurement includes Python/PyTorch imports, deterministic
model construction, opening the copied database, finite-checked adapter
attachment, 16-token continuation, checkpoint/output verification and writing
the small evidence file. It excludes database copying. A nested measurement
records only the first continuation call after attachment. No identity
microbenchmark or redundant standalone identity call is run in the cold worker.
Cold means a new process; operating-system page caches are not flushed. Each
new process must reproduce the baseline complete checkpoint and output hashes.
These single observations describe startup cost and do not establish a cold
speedup. This is the only added confirmation round.
