#!/usr/bin/env python3
"""Summarize retained measurements without rerunning or changing results."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def variants(stage):
    return [stage["same_process"],stage["fresh_process"]["result"],stage["restored_process"]["result"]]


def checks(run):
    return sum(v["checks"] for s in run["stages"] for v in variants(s)) + run["cli"]["checks"] + run["temporal"]["checks"]


def failures(run):
    return sum(len(v["failures"]) for s in run["stages"] for v in variants(s)) + len(run["cli"]["failures"]) + len(run["temporal"]["failures"])


def common_ms(stage,cap=8000):
    return statistics.median(d["cost"]["wall_s"] for d in stage["same_process"]["context_sizes"] if d["cap"]==cap)*1000


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+["| "+" | ".join(map(str,row))+" |" for row in rows])


def main():
    baseline=read("baseline-overlay-results.json")
    final=read("repaired-overlay-results.json")
    baseline_ram=read("baseline-results.json")
    final_ram=read("final-tmpfs-results.json")
    runs=[("Baseline overlay","baseline-overlay-results.json",baseline),
          ("Final overlay","repaired-overlay-results.json",final),
          ("Baseline tmpfs","baseline-results.json",baseline_ram),
          ("Final tmpfs","final-tmpfs-results.json",final_ram)]
    for _,_,r in runs:
        assert not r.get("execution_error"), r.get("execution_error")
        assert failures(r)==0, "A retained workload contains failed assertions"
        assert not r["skipped_stages"]
        assert not r["source_changed_during_run"]
        assert [s["input_sha256"] for s in r["stages"]]==[s["input_sha256"] for s in baseline["stages"]]
        assert r["harness_sha256"]==baseline["harness_sha256"]
    assert final["source_hashes_start"]["store.py"]==final_ram["source_hashes_start"]["store.py"]
    last=final["stages"][-1]
    rows=[]
    for bs,fs in zip(baseline["stages"],final["stages"]):
        rows.append([f"{fs['n_sources']:,}",f"{bs['ingest_cumulative']['wall_s']:.2f} / {fs['ingest_cumulative']['wall_s']:.2f}",
                     f"{bs['same_process']['source_recall_latency']['median_ms']:.3f} / {fs['same_process']['source_recall_latency']['median_ms']:.3f}",
                     f"{common_ms(bs):.2f} / {common_ms(fs):.2f}",f"{fs['sizes_after_checkpoint']['main_bytes']:,}"])
    cost_rows=[]
    for label,filename,r in runs:
        s=r["stages"][-1]
        cost_rows.append([label,f"{r['total']['wall_s']:.2f}",f"{r['total']['parent_cpu_s']+r['total']['child_cpu_s']:.2f}",
                          f"{s['ingest_cumulative']['wall_s']:.2f}",f"{common_ms(s):.2f}",
                          f"{checks(r):,} / {failures(r)}"])
    step_rows=[]
    for key,name in [("backup_cost","Full database backup"),("restore_cost","Verified restore to a new path"),
                     ("export_cost","One-scope JSON export"),("same_process_cost","Complete 50k-stage validation in parent")]:
        step_rows.append([name,f"{last[key]['wall_s']:.3f}",f"{last[key]['cpu_s']:.3f}"])
    for key,name in [("fresh_process","Fresh-process complete validation"),("restored_process","Restored-database complete validation")]:
        step_rows.append([name,f"{last[key]['process_cost']['wall_s']:.3f}","Included in total child CPU"])
    temporal=final["temporal"]
    source_hits=sum(v['exact_source_hits'] for s in final['stages'] for v in variants(s))
    source_total=sum(v['exact_source_total'] for s in final['stages'] for v in variants(s))
    fact_hits=sum(v['current_fact_hits'] for s in final['stages'] for v in variants(s))
    fact_total=sum(v['current_fact_total'] for s in final['stages'] for v in variants(s))
    history_total=sum(v['historical_fact_revisions_checked'] for s in final['stages'] for v in variants(s))
    empty_ms=common_ms(last,256)
    small=[d['actual'] for d in last['same_process']['context_sizes'] if d['cap']==256]
    metadata_hash=final['source_hashes_start']['store.py']
    environment=read('environment.json')
    environment['runs']['repaired-results.json']='tmpfs; intermediate repaired store ce6d26ac, retained; superseded by exact final-hash run'
    environment['runs']['repaired-overlay-pre-retry-results.json']='overlay; intermediate repaired store e4806236, retained; superseded after journal-startup retry refinement'
    environment['runs']['repaired-overlay-results.json']='overlay; final repaired store '+metadata_hash
    environment['runs']['final-tmpfs-results.json']='tmpfs; final repaired store '+metadata_hash
    (HERE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    text=f'''# MEMORY-VALIDATION-002 — archival scale, CLI and temporal evidence

## Result

The registered 1k/10k/50k workload completed against both the original implementation
and the repaired implementation on both available filesystems. Each expanded run
has **{checks(final):,} assertions and {failures(final)} failures**, including **14,577 storage assertions,
43 CLI assertions, and 10 supplied-temporal-reference assertions**. No stage was
skipped, no implementation changed within a run, and all four runs used the same
harness and identical generated operation-stream hashes. These are synthetic
infrastructure checks. No trained writer, reader, temporal binder, or language
model supplied any prediction in this workload.

The broader validation found defects elsewhere; a clean exact-token workload did
not mean the original implementation was correct in every adversarial case.
This report measures the archive behavior and costs of the subsequently repaired
store without substituting those results for model quality.

## Workload and replay accuracy

The final stage contains 50,000 ordinary source messages plus one lexical-control
message, across eight principal/project scopes and 1,600 ordinary conversation
identities (plus the control conversation). Explicit host acceptance creates
1,613 historical fact revisions across 256 current keys: 1,357 revisions are
corrections after the first revision of each key. There are 53,227 mutating API
calls in total: source appends, proposals, acceptances, and the control append.
The live database after checkpoint is {last['sizes_after_checkpoint']['main_bytes']:,} bytes;
the remaining shared-memory sidecar is {last['sizes_after_checkpoint']['sidecar_bytes']['-shm']:,} bytes
and the WAL is truncated to zero. Storage size includes indexes and metadata.
At 50k total messages, each queried scope holds 6,250 ordinary messages; this
workload does not measure a 50k-message single-user/single-project archive.

Across the three stage sizes and three access modes (same process, fresh process,
and fresh process using a restored backup), the final overlay run produced:

* **{source_hits}/{source_total} exact source-retrieval probe instances**. Probes cover early,
  middle, late and seeded-random positions in all scopes. These include repeated
  queries across restarts and are not independent language-understanding examples.
* **{fact_hits}/{fact_total} current-fact recall instances**, with exact value, source and
  revision checks. Every current value also passes a value-query check.
* **{history_total:,} historical-revision checks** against generated values, source contents,
  proposal IDs, scope/key mapping, revision order and host authority. Replays
  repeat histories, so this count does not mean that many distinct revisions.
* **216 foreign-scope probes** returning no source or fact, and **216 rendered
  context-cap checks** within the configured character limits. Scope filtering
  is a local API property; this does not evaluate user authentication.

The literal `car` control is recalled by `car` and missed by `automobile` in every
stage/access mode. This is the predeclared lexical limitation control. Successful
exact-token lookup therefore cannot be described as reliable natural-language
recollection. The source and facts are generated together from known synthetic
inputs; this says nothing about whether a learned model interpreted a sentence
correctly before writing it.

## Primary performance evidence: workspace overlay

Cells showing two measurements list **baseline / final repaired**. Latencies
below are same-process observations, without flushing OS caches. Source latency
uses 48 unique-token probes per stage; the common-word context median uses eight
scopes and a query matching every ordinary observation in its scope.

{table(['Sources','Cumulative ingestion wall, s','Exact-source median, ms','Common-word 8k-char context median, ms','Final DB bytes'],rows)}

The scope-independent ranking repair carries a retrieval cost. At 50k, broad-query
context latency changed from {common_ms(baseline['stages'][-1]):.2f} ms to {common_ms(last):.2f} ms
in the matched overlay runs. The repair's Python lexical scoring examines matching
candidate content; removing foreign-corpus ranking influence does not make that
work free. These single-pass measurements are descriptive observations, not an
isolated statistically controlled speed comparison or an asymptotic guarantee.
Other validation processes could share the host.

The 256-character context cap returned only the {min(small)}-character empty envelope
in every scope at 50k, because no complete source-labelled record fitted. Its
median lookup time was still **{empty_ms:.2f} ms**. A small output cap does not guarantee
that a useful record fits and does not bound the work spent finding candidates.
The 8,000-character cap is a character budget, not a tokenizer or inference budget.

Final 50k-stage extra operations are measured separately below. Backup is the
entire database with all eight scopes; JSON export is only scope zero.

{table(['Operation','Wall seconds','CPU seconds'],step_rows)}

The backup file is {last['backup']['bytes']:,} bytes and the one-scope JSON export is
{last['scope_export']['bytes']:,} bytes. Backups and restored databases share the test
filesystem; this is verified copying and reopening, not independent off-device
disaster recovery. Temporary data files are removed after each run.

## Full workload costs and filesystem qualification

`df -T` identifies `/tmp` as **tmpfs** and the workspace as **overlay**; the raw
command outputs are retained in `environment.json`. tmpfs timing measures a RAM
filesystem, not durable-disk fsync performance. Overlay timing measures the
available container filesystem, not a specified SSD or proof against power loss.
Python is {final['environment']['python'].split()[0]}, SQLite is
{final['environment']['sqlite']}, journal mode is {final['environment']['journal_mode']},
and SQLite synchronous is {final['environment']['synchronous']} (`FULL`).

The full run includes ingestion, oracle checks, recall, context generation,
scope export, backup, restore, fresh-process checks, CLI subprocesses, temporal
reference work and temporary-data cleanup. CPU totals include child-process CPU.

{table(['Run','Total wall, s','Total CPU, s','50k ingestion wall, s','50k common context median, ms','Assertions / failures'],cost_rows)}

For the final overlay run, parent maximum RSS is
{final['total']['parent_max_rss_kib']:,} KiB; the largest child-process reported maximum is
{final['total']['max_child_rss_kib']:,} KiB. These high-water marks are not summed and do not
measure filesystem page cache. The source payloads, SQLite pages, indexes,
backup bytes, generated contexts and interpreter computation all contribute
different costs. No model inference or training cost appears here.

## CLI coverage

Actual subprocesses exercised `record`, `recall`, `propose`, `accept`, `facts`,
`forget`, `forget-session`, `export`, `backup`, and `restore`. Checks cover
cross-invocation retrieval, a revised fact accepted with its expected revision,
authority provenance, cross-project filtering, source erasure, preservation of
the newer independently sourced correction when its older source is erased,
generated IDs, new-target backup/restore behavior, refusal to overwrite existing
backup/restore files, malformed/nonfinite JSON, invalid retrieval limits, explicit
database paths, `COG_MEMORY_PATH`, absolute `XDG_STATE_HOME`, and explicit-path
precedence. These {final['cli']['checks']} assertions passed. This is a local command-line
integration check, not a deployed service, authentication flow or browser test.

## Supplied temporal bindings

A 10,000-edge chain with shuffled claim order and reversed source-map order
remained consistent. Forward and reverse queries returned the expected relation
and all 10,000 provenance sources. A unique selection worked; competing matching
candidates, an unbound alternative, and a cycle did not produce a guessed event.
Invalid source quotations were rejected. Construction took
{temporal['build_cost']['wall_s']*1000:.2f} ms; the forward query took
{temporal['forward_cost']['wall_s']*1000:.2f} ms on the final overlay run.

Twenty bindings were also persisted through ordinary accepted JSON facts and
reloaded in a fresh process. Outer host acceptance alone left the internal
temporal claims `proposed`, so the reference returned `UNKNOWN`. Explicitly
accepting those internal claims after source-span validation yielded `BEFORE`
with all 20 sources. This preserves the two distinct policy decisions. All ten
temporal assertions passed; none learns cue attachment or improves the learned
PHASE-003 selector.

## Reproduction and source identity

Run from `research/nanocog-001`:

```bash
python memory-validation-002/scale/run_scale.py --output /absolute/path/results.json
```

Set `TMPDIR` to an existing directory on the filesystem being measured. The harness
uses bounded temporary databases and deletes them on exit. The registered seed
is `271828`. `NANOCOG_VALIDATION_RUNTIME_ROOT` optionally selects a pinned copy of
the `nanocog` package. The retained result JSON includes every source hash,
environment, operation-stream hash, probe count, failure detail, latency summary,
resource total, and stage-specific file size. The full stage output is larger than
this summary and remains the primary numerical evidence.
The benchmark preserves diagnostic JSON even on a failed assertion or execution
error; successful process exit alone is not its acceptance signal. Inspect the
failure lists and execution-error field. `write_report.py` refuses to summarize
the four named final comparison files as passing if any contains a failure,
skipped stage, mixed source hash, or mismatched generated inputs.

Baseline `store.py` SHA256:
`{baseline['source_hashes_start']['store.py']}`.

Final tested `store.py` SHA256:
`{metadata_hash}`.

The baseline is the published PR head
`56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f`; the repaired working-tree implementation
is pinned by the hashes above and in the JSON, rather than by assuming that a
mutable branch name identifies exact bytes. The final temporal and session files
are unchanged from baseline.

The initial baseline, intermediate repaired tmpfs run and pre-startup-retry
repaired overlay run are retained for audit
but superseded by the four expanded comparison runs listed above. The initial
baseline used the narrower registered first-pass checks; the amendment added
historical value/source checks and explicit CLI correction/filtering before the
paired final runs. The filesystem amendment is recorded in `PROTOCOL.md`.

## Boundaries of the result

This workload establishes tested finite behavior at up to 50k generated messages.
It does not establish lifetime capacity, external backups, failure of the host or
disk, concurrent production load, natural-language memory selection, trained
writer correctness, useful influence on an actual generated answer, privacy
against an adversary controlling the database, or superiority to attention.
Whole-history correctness here means exact storage and retrieval of supplied
structured facts; the earlier learned writer problem remains a separate
qualification task.
'''
    (HERE/'RESULT.md').write_text(text)
    print(json.dumps({"report":str(HERE/'RESULT.md'),"final_assertions":checks(final),"failures":failures(final),
                      "final_store_sha256":metadata_hash,"harness_sha256":final['harness_sha256']},indent=2))


if __name__=='__main__':
    main()
