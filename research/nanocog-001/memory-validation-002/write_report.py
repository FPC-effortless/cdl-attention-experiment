#!/usr/bin/env python3
"""Build the consolidated report from retained evidence; never run a model fit."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "nanocog"


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    suite = read("suite-result.json")
    store = read("storage/results_after_fix.json")
    decoder = read("decoder/final.json")
    system = read("decoder/system_results.json")
    scale = read("scale/repaired-overlay-results.json")
    scale_base = read("scale/baseline-overlay-results.json")
    cost = read("inference/results.json")
    assert suite["result"]["tests"] == "135" and all(suite["result"][k] == "0" for k in ("errors", "failures", "skipped"))
    assert store["counts"] == {"PASS": 9, "FAIL": 0, "ERROR": 0} and store["source_unchanged"]
    assert decoder["tests_run"] == 24 and decoder["success"] and system["passed"] and system["accepted_as_validation"]
    assert not scale["source_changed_during_run"] and not scale["skipped_stages"]
    assert len(cost["cases"]) == 12 and len(cost["cold_processes"]) == 6
    assert cost["harness_sha256"] == hashlib.sha256((ROOT / "inference/run_cost.py").read_bytes()).hexdigest()
    for name, digest in cost["source_sha256"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest
    for cold in cost["cold_processes"]:
        assert cold["revision"] == 2 and cold["tokens_seen"] == cold["history_tokens"] + cold["new_tokens"]
    for stage in scale["stages"]:
        for result in (stage["same_process"], stage["fresh_process"]["result"], stage["restored_process"]["result"]):
            assert not result["failures"]

    names = {"local": "LocalOnly", "packet": "PacketMixer", "typed_packet": "Packet + typed probe"}
    timings = []
    for case in cost["cases"]:
        cells = [names[case["kind"]], str(case["history_tokens"]), str(case["new_tokens"])]
        cells += [f'{case["timings"][path]["median_ms"]:.3f}' for path in
                  ("recompute_prefix", "ram_incremental", "persistent_commit", "exact_retry")]
        timings.append("| " + " | ".join(cells) + " |")
    resources = []
    for model in cost["models"]:
        case = next(c for c in cost["cases"] if c["kind"] == model["kind"] and c["history_tokens"] == 1024 and c["new_tokens"] == 1)
        resources.append(f'| {names[model["kind"]]} | {model["parameters"]:,} | {case["prefix_state"]["raw_tensor_bytes"]:,} | '
                         f'{case["prefix_state"]["encoded_json_bytes"]:,} | {model["standalone_identity"]["median_ms"]:.3f} | '
                         f'{case["standalone_state_codec_round_trip"]["median_ms"]:.3f} |')
    cold_rows = [f'| {names[c["kind"]]} | {c["history_tokens"]} | {c["inner_run_ms"]:.3f} | {c["process_wall_ms"] / 1000:.3f} |'
                 for c in cost["cold_processes"]]
    last = scale["stages"][-1]
    baseline_last = scale_base["stages"][-1]
    broad = lambda stage, cap: statistics.median(c["cost"]["wall_s"] * 1000 for c in stage["same_process"]["context_sizes"] if c["cap"] == cap)
    broad_now, broad_before = broad(last, 8000), broad(baseline_last, 8000)
    text = f'''# MEMORY-VALIDATION-002: what survived the broader tests

23 November 2026 · Engineering validation of draft [PR 94](https://github.com/FPC-effortless/cdl-attention-experiment/pull/94)

## Finding

The stronger tests found real defects in the original memory implementation and
those reproduced defects were repaired. The final package suite passes
**135 tests with no failures or skips**, and the independent process, restart,
restore, archive and capacity checks described below pass. Baseline failures and
intermediate failures are retained beside the final evidence.

**This is not yet a demonstrated automatic lifelong factual-memory model.**
The native decoder restores its computation, but does not automatically consume
the separate semantic ledger. Changing an accepted value from red to blue while
holding native decoder tokens fixed changed no output logits in any of the three
model compositions. The scale test also confirms a lexical limitation: `car`
recalls a record containing `car`; `automobile` does not.

All native model fixtures in this audit use seeded **untrained** weights. The
tests qualify finite engineering behavior, not semantic answer accuracy,
learned writer repair, or a language-model speed/quality advantage. No new model
fit, remote deployment or merge is part of this validation.

## 1. Defects that the original green suite missed

The starting commit is `56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f`.
Its original suite passed 88 tests and skipped one optional archived-source
integration. The independent storage audit initially passed six of nine cases;
the independent decoder audit passed 13 of 19. These are overlapping engineering
views, not a denominator for model accuracy.

| Defect | Reproduction | Repair and observed result |
|---|---|---|
| Relevant facts crowded out | Ten unrelated stock facts from a shared source displaced the correct Orion launch fact from an eight-record context | Direct accepted key/value matches precede source-only fallback; original failure passes |
| Other users influenced ranking | Fifty foreign `beacon` messages changed Alice's ordering for `compass beacon`, without changing Alice's data | Scope-independent lexical coverage/density ranking; another principal or project no longer changes the tested ordering |
| Two startup races | Initializers could reject a peer's new file; the amplified trial later exposed a journal-mode lock conflict | Accept legitimate concurrent creation and bound retries to BUSY/LOCKED journal setup; 20 rounds of six new processes all succeed |
| Incomplete execution identity | Convolution dilation/padding changed computation while preserving weights and shapes | Fingerprint supported execution attributes; live and reopened changes are rejected |
| Unsupported inference hooks | Global hooks changed logits or substituted tokens without changing the old fingerprint | Reject global hooks, class/instance execution replacements after trusted import and compiled call overrides |
| Deletion during first inference | A new checkpoint could commit after source erasure because there was no old checkpoint to invalidate | Capture scope erasure generation before inference and compare inside the commit transaction |
| Unsealed receipt identity | Changing revision 1 to 101 was accepted on replay; valid payload/checksum groups could be moved across identities | Seal metadata together with payload/result digests, validate before use, and re-seal legitimate expiry transitions |

The additional journal-lock failure appeared only after increasing startup
stress. It was preserved and repaired, rather than concealed by the earlier
successful rerun. The system runner now exits nonzero for failures, execution
errors or mixed production sources; CI includes it as a separate process test.

Evidence: [storage audit](storage/REPORT.md), [original storage failures](storage/results_baseline.json),
[journal-lock reproduction](storage/results_startup_lock_reproduction.json),
[final storage run](storage/results_after_fix.json),
[decoder audit](decoder/REPORT.md), [original decoder failures](decoder/baseline.json).

## 2. What the infrastructure actually passed

| Boundary | Executed result |
|---|---|
| Complete package regression | **135/135 passed**, including the hash-verified optional PULSE source integration; no skips |
| Independent process scenarios | **9/9 passed**, including four OS writers plus a reader, conflicts, erasure during response generation and crash boundaries |
| Concurrent first startup | **120/120 successes**, arranged as 20 rounds of six simultaneous processes |
| Process killed before COMMIT | No partial message, search posting or dependency survived |
| Process killed after acknowledgment | The message, posting and dependency survived reopening |
| SQLite capacity exhaustion | Both message insertion and fact replacement rolled back; old values/revisions stayed intact and exact retries succeeded after capacity was restored |
| Derived-data erasure | A 30-link chain, five branches and a saved reply were removed; stale retries failed; another namespace survived |
| Backup and restore | Correct contents and decoder state recovered into new paths; post-erasure backups did not resurrect erased state |
| Schema upgrade | Explicit atomic v1-to-v2 migration, concurrent new-version opens, corrupted-migration rollback and legacy backup restore passed |
| CLI | 43 assertions across record, recall, propose, accept, facts, erasure, export, backup, restore and path configuration |
| Supplied temporal reference | 10 assertions including a 10,000-edge graph and persisted bindings passed; no cue attachment was learned |

The 135-test suite completed in 18.88 seconds in the recorded environment:
Python 3.12.14, SQLite 3.53.1, PyTorch 2.10.0 CPU, NumPy 2.3.5 and pytest 8.4.2.
See [JUnit](full-suite.xml) and [source/environment receipt](suite-result.json).
The PULSE check exercises the pinned archived implementation with synthetic
inputs and forced heads. It is not a rescore of trained PULSE accuracy.

Crash checks using tmpfs establish process-failure semantics. The separate overlay
filesystem measurements establish copying, reopening and the observed costs on
this host. Neither is a physical power-cut or independent-device recovery test.

## 3. Actual decoder continuation

The independent model test uses vocabulary 65, width 128, six layers and a
257-token synthetic sequence, with a fresh interpreter for each of nine uneven
chunks. These model sizes differ from the cost fixtures below.

| Model composition | Parameters | Fresh-process output and full state exact | Restored continuations exact | Max logit difference vs uninterrupted inference |
|---|---:|---:|---:|---:|
| LocalOnly | 919,846 | 9/9 | 4/4 | 8.34e-7 |
| PacketMixer | 1,074,358 | 9/9 | 4/4 | 1.43e-6 |
| Packet + typed probe | 1,218,832 | 9/9 | 4/4 | 9.54e-7 |

All **27 chunk/state comparisons** and **12 backup-restored continuations** were
byte exact against controls with identical chunk boundaries. The uninterrupted
comparison passed `rtol=1e-5, atol=1e-6`, with no changed maximum-logit choices at
any of the 257 positions for any model. Three old-request retries also preserved
the latest state. There were 42 worker invocations.

Resetting state or using a different prefix changed logits and some selected
tokens. This establishes that persisted state affects computation. It does not
establish that the retained information is true, relevant or useful. Crucially,
accepted-fact red/blue swaps made **zero** native-logit difference with fixed
tokens: the host still needs to connect retrieval to its model callback, and a
trained model still needs to use that context correctly.

See [final decoder checks](decoder/final.json) and
[fresh-process model evidence](decoder/system_results.json).

## 4. Archive scale and retrieval cost

The fixed workload contains 50,000 ordinary messages plus one lexical control,
eight principal/project scopes and 1,600 ordinary conversations. Each scope has
6,250 ordinary messages at the final stage, plus the control in scope zero;
this is not 50,000 messages within one scope.
Explicit host acceptance creates 1,613 fact revisions over 256 current keys,
including 1,357 corrections. The model does not generate these values.

Each final expanded run passes **14,630 assertions** across 1k/10k/50k stages,
same-process access, new processes and restored backups. The checks include
432/432 source-retrieval probes, 1,635/1,635 current-fact recall instances and
5,907 historical-revision checks. Repeated replay instances are not independent
language examples. The complete corpus and harness are unchanged across the
baseline and repaired comparison.

At 50k total sources, the workspace-overlay comparison is:

| Measurement | Original | Repaired |
|---|---:|---:|
| Cumulative ingestion | 47.40 s | 54.47 s |
| Exact-source median lookup | 0.200 ms | 0.303 ms |
| Broad common-word context median | {broad_before:.2f} ms | {broad_now:.2f} ms |
| Full workload wall time | 59.69 s | 84.27 s |

The broad-query repair is about **{broad_now / broad_before:.2f} times slower** in
this measured comparison. It scores matching candidate text in Python without
foreign-corpus statistics. The correctness fix does not establish efficiency.
These are descriptive measurements on one shared host, not a controlled
statistical speed claim. The matching tmpfs comparison is separately labelled.

The final database is **44,023,808 bytes**. A full backup took **0.664 seconds**
and verified restore **1.019 seconds**. The 256-character context cap returned an
empty 86-character envelope yet still spent **{broad(last, 256):.2f} ms** median
searching. Bounded output does not imply bounded lookup work or a useful record
that fits. Exact-token success is also compatible with the observed `automobile`
miss for the `car` source.

See [scale report](scale/RESULT.md), [primary final overlay evidence](scale/repaired-overlay-results.json),
[matched baseline](scale/baseline-overlay-results.json) and [filesystem evidence](scale/environment.json).

## 5. What persistence costs during inference

This separate benchmark fixes vocabulary 256, width 160, four layers, seed 91023,
CPU float32 and one Torch thread. Weights and token inputs are identical across
the four execution paths within each model. Model sizes are roughly one million
parameters but are not equal across different mixer compositions. All are
untrained. Other validation workers had finished before the primary timing run.

Each row has one warm-up and five measured trials. Values are **median
milliseconds for the new chunk**. Prefix creation, model construction and session
attachment are excluded from warm timing. Persistent timing includes both model
identity checks, state loading, inference, encoding, SQLite commit and returned
output reconstruction. Retry timing returns an already committed output; it is
not new inference. Raw samples and nearest-rank p95 are retained in the JSON;
with five samples that p95 is merely the sample maximum.

| Model | Prior tokens | New tokens | Recompute full prefix | RAM continuation | Persistent commit | Exact retry |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(timings)}

Use RAM continuation as the direct overhead comparison. A speedup over full-prefix
recomputation does not imply an advantage over an existing recurrent or cached
engine. These fixtures do not compare trained accuracy, attention architectures,
GPU performance or serving throughput.

The following sizes are at 1,024 prior tokens. Raw state includes tensor payloads
only; JSON sizes include codec metadata. These are bounded reference states,
not an unlimited archive of distinguishable semantic facts. An identity check
hashes the model's tensors; the persistent path performs two per successful new
run. Standalone component timings do not sum to total time because the total
includes additional work and execution/cache conditions differ.

| Model | Parameters | Raw state bytes | JSON state bytes | One identity check, ms | State codec round trip, ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(resources)}

Fresh-process results consume only the new 16-token chunk after reopening the
persisted prefix. Process wall time also includes Python/PyTorch startup, model
creation, opening/attachment and output transport; each entry is one observation.

| Model | Prior tokens | Inner persistent call, ms | Whole subprocess, s |
|---|---:|---:|---:|
{chr(10).join(cold_rows)}

All timed paths and six fresh-process results passed the declared logit tolerance
against uninterrupted inference, with expected revisions and token counts.
The actual filesystem is the workspace overlay; this is not a specified physical
SSD benchmark. The primary run took **{cost["wall_seconds"]:.2f} seconds** total.
The initial run is retained; an independent methods review led only to extra
cold-process metadata assertions and a harness source-change guard before the
identical primary repeat. No timing outcome selected a model or configuration.

Evidence: [complete inference measurements](inference/results.json),
[runner](inference/run_cost.py), [initial run](inference/initial-results.json)
and [protocol](PROTOCOL.md).

## 6. Remaining capability and operational boundaries

1. **Automatic factual use is unfinished.** The ledger and the native decoder
   are separate interfaces. `MemorySession` supplies retrieved records to a host
   callback; `PersistentDecoder` consumes supplied tokens only. A learned
   writer-to-ledger-to-answer loop has not been qualified here.
2. **The earlier writer and temporal failures remain open.** Supplied correct
   facts, forced PULSE heads, correct graph traversal and numerical state influence
   do not repair interpretation, commitment, address/value selection, noun copying
   or action–time-cue binding. No fresh-family answer-pair result is supplied.
3. **Retrieval is lexical and broad queries cost more.** Synonyms can miss, and
   searching a large matching candidate set can be costly even for an empty
   bounded output. Whole-history learned accuracy remains unmeasured.
4. **Native state is version-specific.** Actual published-v1 observations and
   facts migrate to SQL v2; old native checkpoints remain stored but attachment
   is rejected under the stronger manifest-v2 identity. A new transient stream
   is required. Stop old writer processes before migration; mixed live binaries
   cannot enforce the new erasure generation.
5. **Erasure and backup have explicit boundaries.** Deletion conservatively
   invalidates native continuation within the scope. It cannot revoke bytes an
   overlapping retry already read. Backups predating deletion retain those data
   until separately erased or reconciled. The host must supply authenticated
   scope identity and durable storage; this is not a deployed multi-user service.
6. **Lifetime reliability is not a finite test result.** Archive and request
   metadata still grow. This study did not test physical power loss, hardware
   endurance, off-device failover, arbitrary future runtime migration or decades
   of operation. CPU reference correctness does not establish those properties.

The present result is a repaired, more thoroughly tested memory runtime with
measured costs and a demonstrated gap at learned factual use. The next meaningful
qualification must connect actual learned proposals, accepted history and answers,
then score full histories and counterfactual answer pairs while preserving
familiar accuracy. Optimizing or enlarging the archive alone cannot close that gap.

## Reproduction and identity

Run the commands in [PROTOCOL.md](PROTOCOL.md) and the three component reports.
The current CI workflow runs the package suite and the independent process/crash
harness; remote status belongs to the actual PR checks. The optional private
PULSE source is not uploaded to CI or to this public repository.

| Production file | SHA-256 |
|---|---|
| store.py | {cost["source_sha256"]["memory/store.py"]} |
| decoder.py | {cost["source_sha256"]["memory/decoder.py"]} |
| codec.py | {cost["source_sha256"]["memory/codec.py"]} |

The consolidated report is generated by `write_report.py`, which verifies the
primary pass records and current source identities before writing this file.
Baseline, intermediate and rejected mixed-source development evidence are
labelled; only the final source-matched runs support the final implementation.
'''
    (ROOT / "RESULT.md").write_text(text)
    print(json.dumps({"report": str(ROOT / "RESULT.md"), "words": len(text.split()), "cases": len(cost["cases"])}))


if __name__ == "__main__":
    main()
