#!/usr/bin/env python3
"""Finite deterministic diagnostic; run from any directory with Python 3.12+."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME_ROOT = Path(os.environ.get("NANOCOG_VALIDATION_RUNTIME_ROOT", str(ROOT)))
sys.path.insert(0, str(RUNTIME_ROOT))
from nanocog.memory.store import MemoryStore
from nanocog.memory.session import MemorySession, context_json
from nanocog.memory.temporal import (
    SourceSpan, TemporalClaim, TemporalReference, BEFORE, AFTER, UNKNOWN,
)

SEED = 271828
STAGES = (1000, 10000, 50000)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: digest(p) for p in sorted((RUNTIME_ROOT / "nanocog/memory").glob("*.py"))}


def scope(index):
    return {"principal_id": f"principal{index // 2}", "namespace": f"project{index % 2}"}


def source_event(i):
    return f"event{i:06d}"


def needle(i):
    return f"needlex{i:06d}"


def source_content(i):
    extra = f" Confirmed label valueprobe{i:06d}." if i % 31 == 0 else ""
    return (f"Synthetic observation {needle(i)}. The archive stores this routine "
            "workplace note in its own conversation. Other unrelated notes must "
            "not change this record or move it into another project's scope." + extra)


def measurements(call):
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    return result, {"wall_s": time.perf_counter()-wall, "cpu_s": time.process_time()-cpu}


def summarize(times):
    ordered = sorted(times)
    return {"n": len(times), "total_ms": sum(times)*1000,
            "median_ms": statistics.median(times)*1000 if times else None,
            "p95_ms": ordered[min(len(ordered)-1, int(.95*(len(ordered)-1)))]*1000 if times else None,
            "max_ms": max(times)*1000 if times else None}


def check(out, name, condition, detail=None):
    out["checks"] += 1
    if not condition:
        out["failures"].append({"check": name, "detail": detail})


def sample_positions(n):
    rng = random.Random(SEED+n)
    positions = list(range(8)) + list(range(n//2-4, n//2+4)) + list(range(n-8,n))
    positions += rng.sample(range(n), 24)
    return sorted(set(positions))


def verify_database(db_path, manifest):
    started = time.perf_counter()
    out = {"checks": 0, "failures": [], "exact_source_hits": 0,
           "exact_source_total": 0, "current_fact_hits": 0, "current_fact_total": 0,
           "latency_s": [], "fact_latency_s": [], "context_sizes": [],
           "scope_isolation_probes": 0, "lexical_control": {},
           "historical_fact_revisions_checked":0}
    stores = [MemoryStore(db_path, **scope(s)) for s in range(8)]
    try:
        for i in manifest["probes"]:
            st = stores[i % 8]
            result, cost = measurements(lambda: st.recall(needle(i), limit=8, max_chars=8000))
            out["latency_s"].append(cost["wall_s"])
            matched = [v for v in result["observations"] if v["event_id"] == source_event(i)]
            good = len(matched) == 1 and matched[0]["content"] == source_content(i)
            out["exact_source_hits"] += good
            out["exact_source_total"] += 1
            check(out, f"exact_source:{i}", good, result if not good else None)
        for i in manifest["probes"][:24]:
            wrong_scope = stores[(i % 8 + 1) % 8]
            found = wrong_scope.recall(needle(i))
            check(out, f"foreign_scope:{i}", not found["facts"] and not found["observations"])
            out["scope_isolation_probes"] += 1
        expected_by_scope = {s: [] for s in range(8)}
        for item in manifest["facts"]:
            expected_by_scope[item["scope"]].append(item)
        for s, expected in expected_by_scope.items():
            actual = {f["key"]: f for f in stores[s].facts()}
            check(out, f"fact_count:scope{s}", len(actual) == len(expected))
            for item in expected:
                found = actual.get(item["key"])
                good = bool(found and found["value"] == item["value"]
                            and found["revision"] == item["revision"]
                            and found["source_event_id"] == item["source_event_id"])
                check(out, f"current_fact:{s}:{item['key']}", good)
                result, cost = measurements(lambda: stores[s].recall(item["key"]))
                out["fact_latency_s"].append(cost["wall_s"])
                good_recall = any(f["key"] == item["key"] and f["value"] == item["value"]
                                  and f["revision"] == item["revision"] for f in result["facts"])
                out["current_fact_hits"] += good_recall
                out["current_fact_total"] += 1
                check(out, f"recalled_fact:{s}:{item['key']}", good_recall)
                value_recall = stores[s].recall(item["value"])
                check(out, f"recalled_current_value:{s}:{item['key']}",
                      any(f["key"] == item["key"] and f["value"] == item["value"]
                          for f in value_recall["facts"]))
                history = stores[s].history(item["key"])
                check(out, f"history:{s}:{item['key']}",
                      [h["revision"] for h in history] == list(range(1,item["revision"]+1)))
                for historical in history:
                    event_number = int(historical["source_event_id"].removeprefix("event"))
                    source = stores[s].get_message(historical["source_event_id"])
                    expected_key = f"factslot{(event_number//(31*8))%32:03d}"
                    historical_good = (historical["value"] == f"valueprobe{event_number:06d}"
                                       and event_number % 31 == 0 and event_number % 8 == s
                                       and historical["key"] == expected_key
                                       and historical["proposal_id"] == f"proposal{event_number}"
                                       and source["content"] == source_content(event_number)
                                       and historical["authority"] == "synthetic-host")
                    check(out, f"history_value_source:{s}:{item['key']}:{historical['revision']}",historical_good)
                    out["historical_fact_revisions_checked"] += 1
                if item["revision"] > 1:
                    obsolete = history[0]["value"]
                    old_recall = stores[s].recall(obsolete)
                    check(out, f"obsolete_value_not_current:{s}:{item['key']}",
                          not any(f["key"] == item["key"] and f["value"] == obsolete
                                  for f in old_recall["facts"]))
            for cap in (256,1024,8000):
                context, context_cost = measurements(lambda:MemorySession(
                    stores[s], session_id="new-check-chat",recall_limit=100,
                    max_context_chars=cap).context("synthetic"))
                size = len(context_json(context))
                out["context_sizes"].append({"scope":s,"cap":cap,"actual":size,
                                             "truncated":context["truncated"],"cost":context_cost})
                check(out, f"context_cap:{s}:{cap}", size <= cap)
                check(out, f"context_labels:{s}:{cap}",
                      all(f["epistemic_status"] == "accepted_by_host" for f in context["accepted_facts"])
                      and all(m["epistemic_status"] == "unverified_observation"
                              for m in context["unverified_observations"]))
        for word in ("car", "automobile"):
            result = stores[0].recall(word)
            out["lexical_control"][word] = any(m["event_id"] == "lexical-control"
                                                     for m in result["observations"])
        check(out, "literal_car_recalled", out["lexical_control"]["car"])
    finally:
        for st in stores:
            st.close()
    out["source_recall_latency"] = summarize(out.pop("latency_s"))
    out["fact_recall_latency"] = summarize(out.pop("fact_latency_s"))
    out["wall_s"] = time.perf_counter() - started
    out["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["source_hashes"] = source_hashes()
    return out


def child_verify(db, manifest_file):
    result, timing = measurements(lambda: subprocess.run(
        [sys.executable, str(__file__), "--verify", str(db), str(manifest_file)],
        cwd=RUNTIME_ROOT, text=True, capture_output=True, timeout=90, check=True))
    return {"process_cost": timing, "result":json.loads(result.stdout)}


def cli_validation(directory):
    out = {"checks":0,"failures":[],"invocations":[],"timing":{}}
    db = directory / "cli.sqlite3"
    env = dict(os.environ)
    env.pop("COG_MEMORY_PATH", None)

    def invoke(args, *, extra=None, env_override=None, code=0):
        full = [sys.executable,"-m","nanocog.memory", "--principal","cli-person",
                "--namespace","cli-project"]
        full += (["--db",str(db)] if extra is None else extra)
        full += args
        done, cost = measurements(lambda: subprocess.run(full, cwd=RUNTIME_ROOT, env=env_override or env,
                                                        text=True,capture_output=True,timeout=20))
        check(out, "cli_exit:"+args[0], done.returncode == code,
              {"code":done.returncode,"stderr":done.stderr} if done.returncode != code else None)
        payload = json.loads(done.stdout if done.returncode == 0 else done.stderr)
        out["invocations"].append({"command":args[0],"exit":done.returncode,"wall_s":cost["wall_s"]})
        return payload

    a = invoke(["record","car violetclitoken","--event-id","cli-source"],extra=["--db",str(db),"--session","cli-chat"])
    check(out,"cli_record_identity",a["session_id"] == "cli-chat" and a["event_id"] == "cli-source")
    p = invoke(["propose","vehicle","\"car\"","cli-source","--proposal-id","cli-proposal"])
    check(out,"cli_proposal_unaccepted",p["status"] == "proposed" and invoke(["facts"]) == [])
    fact = invoke(["accept","cli-proposal","--authority","synthetic-test-host"])
    check(out,"cli_authority",fact["authority"] == "synthetic-test-host")
    found = invoke(["recall","violetclitoken","--limit","8","--max-chars","1024"])
    check(out,"cli_fresh_invocation_recall",any(x["event_id"] == "cli-source" for x in found["unverified_observations"]))
    other = invoke(["recall","violetclitoken"],extra=["--db",str(db),"--namespace","other-project"])
    check(out,"cli_namespace_filter",not other["accepted_facts"] and not other["unverified_observations"])
    invoke(["record","truck yellowclitoken","--event-id","cli-correction"],extra=["--db",str(db),"--session","cli-chat-two"])
    invoke(["propose","vehicle","\"truck\"","cli-correction","--proposal-id","cli-proposal-two","--expected-revision","1"])
    corrected = invoke(["accept","cli-proposal-two","--authority","synthetic-test-host"])
    check(out,"cli_fact_correction",corrected["revision"] == 2 and corrected["value"] == "truck")
    generated = invoke(["record","generated identifiers"])
    check(out,"cli_generated_identity",bool(generated["session_id"] and generated["event_id"]))
    invoke(["propose","bad","NaN","cli-source"],code=2)
    invoke(["recall","car","--limit","0"],code=2)
    exp, backup, restored = directory/"cli.json",directory/"cli-backup.sqlite3",directory/"cli-restored.sqlite3"
    invoke(["export",str(exp)])
    check(out,"cli_export_json",isinstance(json.loads(exp.read_text()),dict))
    invoke(["backup",str(backup)])
    before = digest(backup)
    invoke(["backup",str(backup)],code=2)
    check(out,"cli_backup_no_clobber",digest(backup) == before)
    invoke(["restore",str(backup),str(restored)])
    before = digest(restored)
    invoke(["restore",str(backup),str(restored)],code=2)
    check(out,"cli_restore_no_clobber",digest(restored) == before)
    restored_facts = invoke(["facts","vehicle"],extra=["--db",str(restored)])
    check(out,"cli_restored_fact",restored_facts[0]["value"] == "truck" and restored_facts[0]["revision"] == 2)
    invoke(["forget","cli-source"])
    check(out,"cli_correction_survives_old_source_erasure",invoke(["facts","vehicle"])[0]["value"] == "truck")
    invoke(["forget","cli-correction"])
    check(out,"cli_erased_fact",invoke(["facts","vehicle"]) == [])
    invoke(["record","car violetclitoken","--event-id","cli-source"],
           extra=["--db",str(db),"--session","cli-chat"],code=2)
    invoke(["forget-session",generated["session_id"]])
    env_path = directory/"environment.sqlite3"
    configured = dict(env,COG_MEMORY_PATH=str(env_path))
    invoke(["record","environment path","--event-id","env-source"],extra=[],env_override=configured)
    check(out,"cli_environment_path",env_path.is_file())
    xdg = directory/"xdg"
    configured = dict(env,XDG_STATE_HOME=str(xdg))
    invoke(["record","XDG path","--event-id","xdg-source"],extra=[],env_override=configured)
    check(out,"cli_absolute_xdg_path",(xdg/"nanocog/memory.sqlite3").is_file())
    explicit = directory/"explicit.sqlite3"
    invoke(["record","explicit precedence","--event-id","explicit-source"],
           extra=["--db",str(explicit)],env_override=dict(configured,COG_MEMORY_PATH=str(env_path)))
    check(out,"cli_explicit_path_precedence",explicit.is_file())
    out["timing"]["subprocess_wall_s"] = sum(c["wall_s"] for c in out["invocations"])
    return out


def claim_at(i, *, closing=False):
    action, anchor = f"action{i:05d}", "action00000" if closing else f"action{i+1:05d}"
    text = f"{action} before {anchor}"
    action_span = SourceSpan(0,len(action),action)
    cue_start = len(action)+1
    cue_span = SourceSpan(cue_start,cue_start+6,"before")
    anchor_start = cue_start+7
    anchor_span = SourceSpan(anchor_start,len(text),anchor)
    return text, TemporalClaim(action,BEFORE,anchor,f"temporal-source{i:05d}",
                               cue_span,action_span,anchor_span,status="accepted")


def temporal_validation(directory):
    out = {"checks":0,"failures":[],"edges":10000}
    sources, claims = {},[]
    stream = hashlib.sha256()
    for i in range(10000):
        text, claim = claim_at(i)
        sources[claim.source_event_id] = text
        claims.append(claim)
        stream.update((canonical([text,claim.to_dict()])+"\n").encode())
    random.Random(SEED).shuffle(claims)
    reversed_sources = dict(reversed(list(sources.items())))
    graph, out["build_cost"] = measurements(lambda: TemporalReference(claims,reversed_sources))
    check(out,"long_chain_consistent",graph.consistency.consistent)
    answer, out["forward_cost"] = measurements(lambda: graph.relation_between("action00000","action10000"))
    check(out,"long_chain_forward",answer.relation == BEFORE and len(answer.source_event_ids) == 10000)
    reverse, out["reverse_cost"] = measurements(lambda: graph.relation_between("action10000","action00000"))
    check(out,"long_chain_reverse",reverse.relation == AFTER and len(reverse.source_event_ids) == 10000)
    selected = graph.select_event(["action00000","action10000"],BEFORE,"action05000")
    check(out,"unique_selection",selected.action_id == "action00000")
    check(out,"ambiguous_selection",graph.select_event(["action00000","action00001"],BEFORE,"action10000").reason == "ambiguous")
    check(out,"unbound_alternative",graph.select_event(["action00000","unbound"],BEFORE,"action10000").action_id is None)
    closing_text,closing = claim_at(10000,closing=True)
    cyclic, out["cycle_cost"] = measurements(lambda: TemporalReference(claims+[closing],dict(sources,**{closing.source_event_id:closing_text})))
    check(out,"cycle_unknown",not cyclic.consistency.consistent and cyclic.relation_between("action00000","action00001").relation == UNKNOWN)
    invalid = dict(sources)
    invalid["temporal-source00000"] = "wrong source quotation"
    try:
        TemporalReference(claims,invalid)
        rejected = False
    except ValueError:
        rejected = True
    check(out,"invalid_source_rejected",rejected)
    db = directory/"temporal.sqlite3"
    with MemoryStore(db,principal_id="temporal",namespace="supplied-bindings") as store:
        for i in reversed(range(20)):
            text, claim = claim_at(i)
            # Outer acceptance does not alter the inner claim's own status.
            inner = replace(claim,status="proposed")
            store.append_message(f"temporal-chat{i%3}","user",text,event_id=claim.source_event_id)
            store.propose(f"binding{i:05d}",inner.to_dict(),claim.source_event_id,proposal_id=f"proposal{i}")
            store.accept(f"proposal{i}",authority="synthetic-host")
    done, out["persisted_restart_cost"] = measurements(lambda: subprocess.run(
        [sys.executable,str(__file__),"--temporal-child",str(db)],cwd=RUNTIME_ROOT,text=True,
        capture_output=True,timeout=30,check=True))
    out["restart_result"] = json.loads(done.stdout)
    check(out,"host_acceptance_not_temporal_acceptance",out["restart_result"]["before_inner_acceptance"] == UNKNOWN)
    check(out,"explicit_inner_acceptance_recovers_chain",out["restart_result"]["after_inner_acceptance"] == BEFORE)
    out["input_sha256"] = stream.hexdigest()
    return out


def temporal_child(db):
    with MemoryStore(db,principal_id="temporal",namespace="supplied-bindings") as store:
        facts = store.facts()
        claims = [TemporalClaim.from_dict(f["value"]) for f in facts]
        sources = {c.source_event_id:store.get_message(c.source_event_id)["content"] for c in claims}
        initial = TemporalReference(claims,sources).relation_between("action00000","action00020")
        accepted = [replace(c,status="accepted").validate(sources[c.source_event_id]) for c in claims]
        after = TemporalReference(accepted,sources).relation_between("action00000","action00020")
    return {"before_inner_acceptance":initial.relation,"after_inner_acceptance":after.relation,
            "source_count":len(sources),"accepted_path_source_count":len(after.source_event_ids)}


def file_sizes(db):
    return {"main_bytes":db.stat().st_size,
            "sidecar_bytes":{suffix:Path(str(db)+suffix).stat().st_size
                              if Path(str(db)+suffix).exists() else 0
                              for suffix in ("-wal","-shm","-journal")}}


def run(output):
    started, cpu_start = time.perf_counter(),time.process_time()
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"protocol":"MEMORY-VALIDATION-002-scale", "seed":SEED,
              "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              "environment":{"python":sys.version,"platform":platform.platform(),
                             "sqlite":sqlite3.sqlite_version,"cpu_count":os.cpu_count()},
              "source_hashes_start":source_hashes(),"harness_sha256":digest(__file__),
              "protocol_sha256":digest(HERE/"PROTOCOL.md"),"stages":[],"skipped_stages":[]}
    oracle, stream = {},hashlib.sha256()
    previous = 0
    cumulative_ingest = {"wall_s":0.0,"cpu_s":0.0}
    try:
        with tempfile.TemporaryDirectory(prefix="nanocog-memory-scale-") as scratch:
            directory = Path(scratch)
            db = directory/"scale.sqlite3"
            stores = [MemoryStore(db,**scope(s)) for s in range(8)]
            result["environment"]["journal_mode"] = stores[0].db.execute("PRAGMA journal_mode").fetchone()[0]
            result["environment"]["synchronous"] = stores[0].db.execute("PRAGMA synchronous").fetchone()[0]
            result["environment"]["page_size"] = stores[0].db.execute("PRAGMA page_size").fetchone()[0]
            stores[0].append_message("control-chat","user","My car is in the garage.",event_id="lexical-control")
            for n in STAGES:
                if time.perf_counter()-started > 540 or time.process_time()-cpu_start > 300:
                    result["skipped_stages"].append({"n":n,"reason":"preregistered resource limit"})
                    continue
                wall, cpu = time.perf_counter(),time.process_time()
                operation_count = 0
                for i in range(previous,n):
                    s = i%8
                    session_id = f"conversation{(i//8)%200:03d}"
                    content = source_content(i)
                    stores[s].append_message(session_id,"user",content,event_id=source_event(i))
                    stream.update((canonical(["record",s,session_id,source_event(i),content])+"\n").encode())
                    operation_count += 1
                    if i%31 == 0:
                        key = f"factslot{(i//(31*8))%32:03d}"
                        current = oracle.get((s,key))
                        revision = current["revision"] if current else 0
                        value = f"valueprobe{i:06d}"
                        stores[s].propose(key,value,source_event(i),proposal_id=f"proposal{i}",expected_revision=revision)
                        stores[s].accept(f"proposal{i}",authority="synthetic-host")
                        stream.update((canonical(["fact",s,key,value,source_event(i),revision])+"\n").encode())
                        operation_count += 2
                        oracle[(s,key)] = {"scope":s,"key":key,"value":value,"revision":revision+1,
                                           "source_event_id":source_event(i)}
                cost = {"wall_s":time.perf_counter()-wall,"cpu_s":time.process_time()-cpu}
                for key in cumulative_ingest:
                    cumulative_ingest[key] += cost[key]
                stage = {"n_sources":n,"source_control_records":1,"new_sources":n-previous,
                         "new_mutating_calls":operation_count,"ingest_increment":cost,
                         "ingest_cumulative":dict(cumulative_ingest),"input_sha256":stream.hexdigest(),
                         "accepted_current_facts":len(oracle),"accepted_fact_revisions":sum(v["revision"] for v in oracle.values()),
                         "sizes_before_checkpoint":file_sizes(db)}
                manifest = {"n":n,"probes":sample_positions(n),"facts":list(oracle.values())}
                manifest_path = directory/f"manifest-{n}.json"
                manifest_path.write_text(canonical(manifest))
                stage["manifest_sha256"] = digest(manifest_path)
                stage["same_process"],stage["same_process_cost"] = measurements(lambda:verify_database(db,manifest))
                # No parent connection survives the reopen diagnostic.
                for st in stores:
                    st.close()
                stage["fresh_process"] = child_verify(db,manifest_path)
                stores = [MemoryStore(db,**scope(s)) for s in range(8)]
                backup = directory/f"backup-{n}.sqlite3"
                stage["backup"],stage["backup_cost"] = measurements(lambda:stores[0].backup(backup))
                export = directory/f"export-{n}.json"
                stage["scope_export"],stage["export_cost"] = measurements(lambda:stores[0].export_json(export))
                # Scope export is deliberately one of eight scopes, backup is all.
                exported = json.loads(export.read_text())
                stage["export_top_keys"] = sorted(exported)
                restored = directory/f"restored-{n}.sqlite3"
                stage["restore"],stage["restore_cost"] = measurements(lambda:MemoryStore.restore_backup(backup,restored))
                stage["restored_process"] = child_verify(restored,manifest_path)
                stores[0].db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                stage["sizes_after_checkpoint"] = file_sizes(db)
                stage["source_hashes"] = source_hashes()
                stage["parent_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                result["stages"].append(stage)
                output.write_text(json.dumps(result,indent=2)+"\n")
                failures = sum(len(c["failures"]) for c in (stage["same_process"],stage["fresh_process"]["result"],stage["restored_process"]["result"]))
                print(f"STAGE n={n} ingestion_wall={cost['wall_s']:.3f}s failures={failures} database_bytes={stage['sizes_after_checkpoint']['main_bytes']}",flush=True)
                previous = n
            for st in stores:
                st.close()
            result["cli"],result["cli_cost"] = measurements(lambda:cli_validation(directory))
            print(f"CLI checks={result['cli']['checks']} failures={len(result['cli']['failures'])}",flush=True)
            result["temporal"],result["temporal_cost"] = measurements(lambda:temporal_validation(directory))
            print(f"TEMPORAL checks={result['temporal']['checks']} failures={len(result['temporal']['failures'])}",flush=True)
    except Exception as exc:
        result["execution_error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        print(result["execution_error"]["traceback"],file=sys.stderr,flush=True)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["total"] = {"wall_s":time.perf_counter()-started,"parent_cpu_s":time.process_time()-cpu_start,
                       "child_cpu_s":children.ru_utime+children.ru_stime-child_cpu_start.ru_utime-child_cpu_start.ru_stime,
                       "parent_max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "max_child_rss_kib":children.ru_maxrss}
    result["source_hashes_end"] = source_hashes()
    result["source_changed_during_run"] = result["source_hashes_start"] != result["source_hashes_end"]
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results.json")
    parser.add_argument("--verify",nargs=2)
    parser.add_argument("--temporal-child")
    args = parser.parse_args()
    if args.verify:
        print(canonical(verify_database(Path(args.verify[0]),json.loads(Path(args.verify[1]).read_text()))))
    elif args.temporal_child:
        print(canonical(temporal_child(Path(args.temporal_child))))
    else:
        run(args.output)
