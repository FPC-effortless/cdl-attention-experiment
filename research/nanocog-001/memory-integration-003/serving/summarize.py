#!/usr/bin/env python3
"""Rebuild the serving evidence tables from retained observations."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
KINDS = ("local", "packet", "typed_packet")
LABELS = {"local": "Local", "packet": "Packet", "typed_packet": "Packet + typed"}


def read(path):
    return json.loads(path.read_text())


def percent(old, new):
    return (old - new) / old * 100


rows = []
tables = {}
all_runs = {}
for label, directory in (("initial", HERE), ("confirmation", HERE / "confirmation")):
    table = ["| Model | History + new tokens | Baseline, ms | Candidate, ms | Cost reduction |",
             "|---|---:|---:|---:|---:|"]
    all_runs[label] = {}
    for kind in KINDS:
        old = read(directory / f"{kind}-baseline.json")
        new = read(directory / f"{kind}-candidate.json")
        all_runs[label][kind] = (old, new)
        assert old["manifest"] == new["manifest"]
        for run in (old, new):
            assert run["source_sha256"] == run["source_sha256_after"]
        for name in old["source_sha256"]:
            if name not in ("memory/decoder.py", "memory/codec.py"):
                assert old["source_sha256"][name] == new["source_sha256"][name]
        if label == "confirmation":
            initial_old, initial_new = all_runs["initial"][kind]
            assert old["source_sha256"] == initial_old["source_sha256"]
            assert new["source_sha256"] == initial_new["source_sha256"]
        for a, b in zip(old["cases"], new["cases"]):
            assert (a["history_tokens"], a["new_tokens"]) == (b["history_tokens"], b["new_tokens"])
            assert a["checkpoint_sha256"] == b["checkpoint_sha256"]
            assert a["output_sha256"] == b["output_sha256"]
            ta, tb = a["timings"]["persistent"]["median_ms"], b["timings"]["persistent"]["median_ms"]
            if label == "confirmation":
                assert tb < ta
            table.append(f"| {LABELS[kind]} | {a['history_tokens']} + {a['new_tokens']} | {ta:.3f} | {tb:.3f} | {percent(ta, tb):.1f}% |")
            for metric in ("ram", "persistent", "retry"):
                for variant, case in (("baseline", a), ("candidate", b)):
                    timing = case["timings"][metric]
                    rows.append(dict(round=label, model=kind, history_tokens=case["history_tokens"],
                                     new_tokens=case["new_tokens"], variant=variant, metric=metric,
                                     median_ms=timing["median_ms"], sample_max_p95_ms=timing["p95_ms"],
                                     samples_ms=json.dumps(timing["samples_ms"])))
    tables[label] = "\n".join(table)

with (HERE / "timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

confirmation = read(HERE / "confirmation/comparison.json")
initial = read(HERE / "comparison.json")
assert initial["passed"] and confirmation["passed"]
identity = ["| Model | History | Initial baseline → candidate, ms | Confirmation baseline → candidate, ms |",
            "|---|---:|---:|---:|"]
for kind in KINDS:
    for history in (128, 1024):
        measures = []
        for label in ("initial", "confirmation"):
            pair = all_runs[label][kind]
            measures.append([next(c for c in run["cases"] if c["history_tokens"] == history)["hot_identity"]["median_ms"] for run in pair])
        identity.append(f"| {LABELS[kind]} | {history} | {measures[0][0]:.3f} → {measures[0][1]:.3f} | {measures[1][0]:.3f} → {measures[1][1]:.3f} |")

ram = ["| Model | History + new tokens | RAM baseline / candidate, ms | Durable candidate, ms | Retry baseline / candidate, ms |",
       "|---|---:|---:|---:|---:|"]
for kind in KINDS:
    for a, b in zip(*[run["cases"] for run in all_runs["confirmation"][kind]]):
        ra, rb = [c["timings"]["ram"]["median_ms"] for c in (a, b)]
        rta, rtb = [c["timings"]["retry"]["median_ms"] for c in (a, b)]
        persistent = b["timings"]["persistent"]["median_ms"]
        ram.append(f"| {LABELS[kind]} | {a['history_tokens']} + {a['new_tokens']} | {ra:.3f} / {rb:.3f} | {persistent:.3f} | {rta:.3f} / {rtb:.3f} |")

cold = ["| Model | New-process wall baseline / candidate, ms | First continuation baseline / candidate, ms |",
        "|---|---:|---:|"]
for result in confirmation["models"]:
    a, b = [result["cold"][variant] for variant in ("baseline", "candidate")]
    cold.append(f"| {LABELS[result['kind']]} | {a['process_wall_ms']:.3f} / {b['process_wall_ms']:.3f} | {a['inner_run_ms']:.3f} / {b['inner_run_ms']:.3f} |")

verification = ET.parse(HERE / "verification.xml").getroot()
suites = list(verification) if verification.tag == "testsuites" else [verification]
tests = sum(int(s.attrib["tests"]) for s in suites)
failures = sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites)
skipped = sum(int(s.attrib["skipped"]) for s in suites)
assert tests == 55 and failures == skipped == 0
hashes = read(HERE / "verification_sources.json")
assert hashes["before"] == hashes["after"]
for name in ("memory/decoder.py", "memory/codec.py"):
    assert all_runs["confirmation"]["local"][1]["source_sha256"][name] == hashes["after"]["nanocog/" + name]
source_table = ["| File | SHA-256 |", "|---|---|"]
source_table += [f"| `{name}` | `{digest}` |" for name, digest in hashes["after"].items()]

report = f"""# Native decoder serving optimization

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

Correctness passed at three levels: {tests} decoder/adversarial/optimization
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

Baseline: published commit `{initial['baseline_commit']}`. A git archive was
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
{confirmation['filesystem']}
```

## All initial warm persistent results

Numbers are medians in milliseconds. Positive reduction means lower candidate
cost; a negative value means a measured regression.

{tables['initial']}

The initial Local 128 + 1 and Packet 1,024 + 1 cases regressed. The unchanged
Packet 1,024 + 1 RAM control also shifted from about 4.930 to 8.715 ms, so that
pass cannot support a stable across-case latency claim by itself.

## All quiet-confirmation warm persistent results

{tables['confirmation']}

Each listed median decreased, but the improvement size depends on model and
chunk/history length. Five observations per case on one shared host are not a
confidence interval or a universal performance guarantee. The independent
identity microbenchmark and exact-byte equivalence checks support the proposed
mechanism rather than relying on the favorable timing round alone.

## Full model-identity recheck cost

These medians measure one hot `_check_model` call. They are measured separately
and must not be added to, or subtracted from, the end-to-end timings as if the
samples were identical. Attachment still includes finite scanning.

{chr(10).join(identity)}

## RAM and receipt replay controls

These are all quiet-confirmation cases. Both RAM controls are shown to expose
host/process variation even though their implementation and weights are equal.
Durable calls remain more expensive than a RAM-only call, and receipt retries
still perform identity and persisted-receipt verification.

{chr(10).join(ram)}

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

{chr(10).join(cold)}

The typed candidate was slower in this single cold observation, both for total
process time and for the first continuation. The cold results therefore do not
support a claim of consistent startup improvement.

The first pass also tested each candidate directly against a baseline native
checkpoint. All three unchanged v2 manifests matched, and output/checkpoint
bytes matched. Thus compatibility is checked using old persisted sessions,
not merely by comparing two newly created candidate sessions.

## Integrated correctness and source freeze

The final run against the integrated store passed **{tests}/{tests} tests**, with
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

{chr(10).join(source_table)}

The initial harness is retained as `initial_compare.py` with SHA-256
`{initial['harness_sha256']}`. The confirmation harness is `compare.py`, SHA-256
`{confirmation['harness_sha256']}`. Each worker JSON records package hashes,
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
"""
(HERE / "RESULT.md").write_text(report)
print(json.dumps(dict(report=str(HERE / "RESULT.md"), timing_rows=len(rows), tests=tests)))
