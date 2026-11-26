# Native decoder serving optimization

Evidence: [prospective protocol](PROTOCOL.md), [initial comparison](comparison.json),
[quiet confirmation](confirmation/comparison.json), [all timing samples](timings.csv),
[55-case test result](verification.xml), and [integrated source freeze](verification_sources.json).
The [initial harness](initial_compare.py), [confirmation harness](compare.py),
and [table generator](summarize.py) are retained.

## Finding

The optimized decoder preserves the published manifest and checkpoint bytes
while reducing redundant identity-check work. Every one of the 12 prespecified
warm persistent-call medians improved in the single quiet confirmation. The
initial pass improved 10 of 12 medians, with two regressions and substantial
variation in unchanged RAM controls. Both complete passes are retained below.
These measurements support a local engineering improvement; they do not prove
general serving efficiency or learned-model accuracy.

Correctness passed at three levels: 55 decoder/adversarial/optimization
tests against the final integrated store, exact output and complete-checkpoint
hash equality across all 12 baseline/candidate cases in each timing pass, and
continuation of all three model types from published-baseline native sessions.
No existing decoder manifest or wire format was changed.

## What changed, and why the guard remains intact

`codec._tensor_buffer` exposes a canonical contiguous CPU byte view to hashlib.
For an already contiguous CPU tensor this avoids the copied tensor-byte buffer.
Noncontiguous inputs are first made contiguous; GPU inputs would still require
a CPU transfer. State serialization still calls the finite-checked `tensor_bytes`
function and produces owned bytes. Incremental hash updates for header, separator
and tensor contents produce the same byte stream as the former concatenation.

Model attachment still checks every floating tensor for finite values. Before
and after each new inference, and before a receipt replay, the adapter still
hashes every model parameter/buffer byte and compares the complete manifest to
the attachment identity. Only the redundant finite scan is omitted on those
rechecks. Under the same SHA-256 collision assumption used by the existing
integrity contract, content equality to the previously checked finite tensors
implies unchanged finite values. A change to NaN, infinity or any ordinary value
changes the identity and is rejected. No tensor-version counter is substituted
for byte checks; the regression suite explicitly changes `.data` without
incrementing `_version` and verifies rejection before committing.

One live named-module traversal is reused within each identity call. Attribute
presence is checked against current instance and class dictionaries before
normal attribute access, avoiding exceptions for thousands of absent geometry
attributes. Descriptors and newly introduced class attributes remain live and
have explicit mutation regressions. Implementation files are still read and
hashed on each check; no implementation-byte cache was introduced.

Receipt seals and payload/result checks, scope and chronology validation,
shape/dtype/resource limits, erasure-generation checks, expected revisions,
output reconstruction and synchronous commit-before-return semantics were
left in place. No checkpoint or receipt write was omitted or deferred.

## Experimental design

Baseline: published commit `ffb842adc176496d8838b774d0fe412aec5d1d20`. A git archive was
frozen before edits, then copied to a separate candidate package. Only the
candidate decoder and codec were replaced. Both timed variants use the same
published store, separating this measurement from the archive retrieval change.
The final integrated-store checks are reported separately below.

The fixtures are seeded, untrained reference decoders: LocalOnly has 1,005,474
parameters, PacketMixer 1,134,210, and PacketAndTypedProbe 1,274,942. All use
vocabulary 256, width 160, four layers, batch one, float32, one Torch thread and
seed 91023. Tokens use seed 91024. Training, model quality and GPU performance
are outside this experiment.

Each warm case has one discarded warm-up and five measured trials. Prefix
lengths are 128 and 1,024; new chunks have one or 16 tokens. Store opening,
adapter attachment, database copying and prefix construction occur outside the
warm call timer. The persistent timer includes the complete `.run` path:
fingerprint checks, state loading and validation, inference, serialization,
SQLite commit, and returned-output reconstruction. Retry timing includes
identity and receipt validation, and must reproduce the exact logits without
changing the checkpoint. The RAM control starts with an already computed
in-memory prefix state. Every raw sample and sample-maximum p95 is in the JSON
evidence and `timings.csv`; with five samples, p95 is the maximum observation,
not a precise estimate of a population tail.

The first process order was baseline/candidate for LocalOnly, candidate/baseline
for PacketMixer, and baseline/candidate for the typed model. A single quiet
confirmation was authorized and prospectively recorded in `PROTOCOL.md` after
the first pass showed timing noise. Other agents paused heavy work. The second
pass reversed each order and retained all other cases and warm measurements.
The production files did not change between these passes. No further timing
round was selected based on a favorable result.

Filesystem recorded by the confirmation:

```text
Filesystem     Type    1024-blocks     Used Available Capacity Mounted on
overlay        overlay    66002476 20978988  41651664      34% /workspace
```

## All initial warm persistent results

Numbers are medians in milliseconds. Positive reduction means lower candidate
cost; a negative value means a measured regression.

| Model | History + new tokens | Baseline, ms | Candidate, ms | Cost reduction |
|---|---:|---:|---:|---:|
| Local | 128 + 1 | 29.047 | 37.825 | -30.2% |
| Local | 128 + 16 | 34.135 | 23.859 | 30.1% |
| Local | 1024 + 1 | 51.399 | 41.450 | 19.4% |
| Local | 1024 + 16 | 53.242 | 42.528 | 20.1% |
| Packet | 128 + 1 | 41.805 | 25.621 | 38.7% |
| Packet | 128 + 16 | 54.657 | 38.837 | 28.9% |
| Packet | 1024 + 1 | 57.214 | 78.500 | -37.2% |
| Packet | 1024 + 16 | 101.087 | 56.689 | 43.9% |
| Packet + typed | 128 + 1 | 60.405 | 40.203 | 33.4% |
| Packet + typed | 128 + 16 | 91.689 | 72.828 | 20.6% |
| Packet + typed | 1024 + 1 | 83.340 | 62.373 | 25.2% |
| Packet + typed | 1024 + 16 | 126.992 | 77.810 | 38.7% |

The initial Local 128 + 1 and Packet 1,024 + 1 cases regressed. The unchanged
Packet 1,024 + 1 RAM control also shifted from about 4.930 to 8.715 ms, so that
pass cannot support a stable across-case latency claim by itself.

## All quiet-confirmation warm persistent results

| Model | History + new tokens | Baseline, ms | Candidate, ms | Cost reduction |
|---|---:|---:|---:|---:|
| Local | 128 + 1 | 28.767 | 15.718 | 45.4% |
| Local | 128 + 16 | 28.911 | 24.359 | 15.7% |
| Local | 1024 + 1 | 46.815 | 31.733 | 32.2% |
| Local | 1024 + 16 | 47.228 | 33.800 | 28.4% |
| Packet | 128 + 1 | 32.373 | 24.741 | 23.6% |
| Packet | 128 + 16 | 45.091 | 36.028 | 20.1% |
| Packet | 1024 + 1 | 48.091 | 46.530 | 3.2% |
| Packet | 1024 + 16 | 57.936 | 50.260 | 13.2% |
| Packet + typed | 128 + 1 | 53.208 | 32.647 | 38.6% |
| Packet + typed | 128 + 16 | 68.862 | 64.827 | 5.9% |
| Packet + typed | 1024 + 1 | 73.512 | 45.449 | 38.2% |
| Packet + typed | 1024 + 16 | 89.125 | 74.490 | 16.4% |

Each listed median decreased, but the improvement size depends on model and
chunk/history length. Five observations per case on one shared host are not a
confidence interval or a universal performance guarantee. The independent
identity microbenchmark and exact-byte equivalence checks support the proposed
mechanism rather than relying on the favorable timing round alone.

## Full model-identity recheck cost

These medians measure one hot `_check_model` call. They are measured separately
and must not be added to, or subtracted from, the end-to-end timings as if the
samples were identical. Attachment still includes finite scanning.

| Model | History | Initial baseline → candidate, ms | Confirmation baseline → candidate, ms |
|---|---:|---:|---:|
| Local | 128 | 10.450 → 6.183 | 9.688 → 4.376 |
| Local | 1024 | 17.671 → 6.511 | 9.477 → 4.503 |
| Packet | 128 | 11.648 → 5.947 | 18.512 → 5.455 |
| Packet | 1024 | 12.405 → 6.800 | 10.725 → 6.228 |
| Packet + typed | 128 | 17.273 → 7.401 | 16.293 → 10.091 |
| Packet + typed | 1024 | 16.911 → 9.238 | 13.385 → 6.906 |

## RAM and receipt replay controls

These are all quiet-confirmation cases. Both RAM controls are shown to expose
host/process variation even though their implementation and weights are equal.
Durable calls remain more expensive than a RAM-only call, and receipt retries
still perform identity and persisted-receipt verification.

| Model | History + new tokens | RAM baseline / candidate, ms | Durable candidate, ms | Retry baseline / candidate, ms |
|---|---:|---:|---:|---:|
| Local | 128 + 1 | 1.772 / 1.325 | 15.718 | 10.211 / 4.782 |
| Local | 128 + 16 | 2.643 / 2.428 | 24.359 | 11.197 / 5.852 |
| Local | 1024 + 1 | 1.452 / 1.400 | 31.733 | 10.034 / 4.735 |
| Local | 1024 + 16 | 2.347 / 1.991 | 33.800 | 10.910 / 5.016 |
| Packet | 128 + 1 | 2.622 / 2.707 | 24.741 | 11.042 / 6.337 |
| Packet | 128 + 16 | 11.925 / 12.277 | 36.028 | 11.888 / 7.054 |
| Packet | 1024 + 1 | 2.802 / 3.184 | 46.530 | 11.010 / 6.797 |
| Packet | 1024 + 16 | 11.531 / 12.972 | 50.260 | 11.645 / 6.313 |
| Packet + typed | 128 + 1 | 5.263 / 7.791 | 32.647 | 17.638 / 7.970 |
| Packet + typed | 128 + 16 | 30.008 / 34.099 | 64.827 | 16.930 / 8.382 |
| Packet + typed | 1024 + 1 | 5.648 / 4.582 | 45.449 | 19.354 / 7.502 |
| Packet + typed | 1024 + 16 | 28.910 / 28.379 | 74.490 | 15.107 / 7.811 |

## Cold process and existing-session compatibility

Exactly one new-process observation per model and variant was added in the
prospective confirmation amendment. Both variants opened a copy of the baseline
1,024-token native checkpoint and continued 16 tokens. Every resulting complete
checkpoint and output hash matched the baseline warm result.

The process wall boundary includes Python/PyTorch imports, model construction,
database opening, finite-checked attachment, continuation, hash verification and
writing the small evidence file. Database copying is outside the timer. The
inner timing includes only the first continuation after attachment. Operating
system page caches were not flushed. These are descriptive startup observations,
with no claim that a single sample establishes a cold-speed improvement.

| Model | New-process wall baseline / candidate, ms | First continuation baseline / candidate, ms |
|---|---:|---:|
| Local | 2130.098 / 2088.671 | 55.291 / 43.370 |
| Packet | 2229.688 / 2175.382 | 103.223 / 61.058 |
| Packet + typed | 2382.763 / 2762.137 | 107.231 / 157.769 |

The typed candidate was slower in this single cold observation, both for total
process time and for the first continuation. The cold results therefore do not
support a claim of consistent startup improvement.

The first pass also tested each candidate directly against a baseline native
checkpoint. All three unchanged v2 manifests matched, and output/checkpoint
bytes matched. Thus compatibility is checked using old persisted sessions,
not merely by comparing two newly created candidate sessions.

## Integrated correctness and source freeze

The final run against the integrated store passed **55/55 tests**, with
zero failures, errors or skips. `verification.xml` records the test cases.
Existing tests cover reference continuation, receipt idempotence/conflicts,
limits, scope/manifest mismatch, corrupt state and receipts, erasure races,
execution hooks and mutable execution identity. The new 15 cases cover explicit
canonical bytes, noncontiguous buffers and lifetime, `.data` mutation including
NaN/Inf, initial finite checks, mutation during inference, descriptor changes,
live class attributes and equivalent finite manifests.

The integrated store includes the independently measured exact tokenization
optimization and the fact-snapshot/CAS API; it is not the timed frozen store.
Source hashes were captured before and after the regression run and were equal:

| File | SHA-256 |
|---|---|
| `nanocog/memory/codec.py` | `195758561841b95cb67046b23a8e462f77fc47f36d7f5a988fd0601b4f0e23ff` |
| `nanocog/memory/decoder.py` | `8380db65776f50329cfa15575211e7b72e2dc9629799174bc345addbd6f6b6e2` |
| `nanocog/memory/store.py` | `89dc05ae971c52b73b65bf37df1349ef16473528d1f4a8c695a4c891f0ea6085` |
| `nanocog/memory/test_adversarial_decoder.py` | `71019a82bd98e5e911d6b81e96eda1c7e30867f4d545be22c3998439140a5f52` |
| `nanocog/memory/test_decoder.py` | `0a4b5dbb7446b64008d131ac6a0412ef9869f208fffd877ba01ebdbb55f8bb56` |
| `nanocog/memory/test_serving_optimization.py` | `88826be705e98abd26da8dba15123db0b4dc5ed1fe206141f80e71d0fa1b0a9d` |

The initial harness is retained as `initial_compare.py` with SHA-256
`86a6a33d8b38c904be2b4b10d5a97bc70433ae0f494474003fc8cf20b26fc73d`. The confirmation harness is `compare.py`, SHA-256
`f24b2f05c57de205542294501c1da6717e8188aaa4f546a3ee59f14a8834533c`. Each worker JSON records package hashes,
Python/Torch/SQLite versions, manifests, output/checkpoint hashes and every
timing observation. `comparison.json` and `confirmation/comparison.json` record
the corresponding comparison gates.

## Remaining cost and limits

Full content hashing is intentionally still proportional to model bytes and
occurs twice on a new inference. This experiment does not make mutable-model
identity validation constant time. Prefix length changes the size of the prior
saved result that the current store verifies; prior result parsing and duplicate
checkpoint reads remain measurable overhead. They were retained rather than
changing the receipt/integrity contract within this optimization.

This establishes equal seeded weights, exact persisted bytes and equivalent
reference computation; it does not qualify a trained writer, familiar accuracy,
whole-history semantic correctness or counterfactual factual-answer quality.
Those are separate gates in the learned-memory experiment. Histories beyond
1,024 tokens, many concurrent clients, real hardware durability under power
failure, production-trained model sizes and lifetime retention were not measured
by this serving-cost comparison. The justified claim is narrower: redundant
identity work was removed without a regression in the tested contract, and the
prespecified quiet local comparison reduced all measured warm-call medians.

## Reproduction

Extract the published baseline package into the sibling
`memory003_serving_baseline/` directory, preserving its
`research/nanocog-001/nanocog` path. Run `compare.py --output-dir <new-directory>`
from this checkout; it rejects an existing output directory to avoid overwriting
evidence. It copies that baseline package and overlays only the current decoder
and codec. Run the three test modules named in `verification.xml`, then regenerate
the report with `summarize.py`. Reproduction on another machine is a new
observation, not an expectation of these exact latency values.
