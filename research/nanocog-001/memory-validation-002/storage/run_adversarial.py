"""Bounded adversarial storage experiments, using synthetic temporary data only.

This is an independent system validation harness, not a model evaluation. Run:
    python research/nanocog-001/memory-validation-002/storage/run_adversarial.py
Its default JSON report is adjacent to this file. All databases are temporary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanocog.memory.store import MemoryStore, ConflictError, DeletedError
from nanocog.memory.session import MemorySession

CTX = mp.get_context("spawn")


def accepted(store, key, value, source, proposal, revision=0):
    store.propose(key, value, source, proposal_id=proposal, expected_revision=revision)
    return store.accept(proposal, authority="synthetic-fixture")


def exception_name(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return None


def ranking_interference(root):
    """Foreign text must not change this user's otherwise-identical retrieval."""
    variants = []
    for variant in ("foreign_principal", "foreign_namespace"):
        path = root / (variant + ".sqlite3")
        with MemoryStore(path, principal_id="alice", namespace="home") as own:
            own.append_message("a", "user", "compass", event_id="a")
            own.append_message("a", "user", "beacon", event_id="b")
            before = [x["event_id"] for x in own.messages(query="compass beacon")]
            foreign_scope = dict(principal_id="bob", namespace="home") if variant == "foreign_principal" else dict(principal_id="alice", namespace="work")
            with MemoryStore(path, **foreign_scope) as other:
                for i in range(50):
                    other.append_message("foreign", "user", "beacon foreigncanary", event_id=f"f{i}")
            after = [x["event_id"] for x in own.messages(query="compass beacon")]
            leaked = own.recall("foreigncanary")["observations"]
            variants.append(dict(variant=variant, before=before, after=after,
                                 unchanged=before == after, foreign_content_rows=len(leaked)))
    return dict(passed=all(x["unchanged"] and not x["foreign_content_rows"] for x in variants),
                expectation="Foreign-scope writes cannot change local retrieval ranking or return foreign content.",
                observations=variants)


def unrelated_facts(root):
    path = root / "irrelevant.sqlite3"
    with MemoryStore(path, principal_id="alice") as store:
        content = "Orion launches Thursday. " + " ".join(f"Stock item {i} has {10+i} units." for i in range(10))
        store.append_message("source-chat", "user", content, event_id="source")
        accepted(store, "orion.launch", "Thursday", "source", "target")
        session = MemorySession(store, session_id="new-chat")
        before = session.context("When does Orion launch?")
        for i in range(10):
            accepted(store, f"stock.{i}", 10+i, "source", f"stock-p{i}")
        after = session.context("When does Orion launch?")
        stored_target = store.facts("orion.launch")
        target_in_after = any(x["key"] == "orion.launch" for x in after["accepted_facts"])
        return dict(passed=target_in_after, expectation="A directly query-matching accepted fact stays available after unrelated facts are added to its source.",
                    before=before, after=after, stored_target=stored_target,
                    unrelated_writes=10, ledger_corrupted=False)


def process_writer(path, worker, ready, start, output, count):
    try:
        ready.put(worker)
        start.wait(10)
        t0 = time.perf_counter()
        with MemoryStore(path, principal_id="shared", timeout=15) as store:
            for i in range(count):
                event = f"worker-{worker}-event-{i}"
                store.append_message(f"chat-{worker}", "user", f"worker{worker} item{i} value{worker*10000+i}", event_id=event)
                accepted(store, f"worker{worker}.item{i}", worker*10000+i, event, f"p-{worker}-{i}")
                # Both ordinary and FTS reads run while peer processes write.
                if store.facts(f"worker{worker}.item{i}")[0]["value"] != worker*10000+i:
                    raise AssertionError("writer read back a wrong value")
                if not store.messages(query=f"value{worker*10000+i}"):
                    raise AssertionError("acknowledged message missing from FTS")
        output.put(dict(worker=worker, ok=True, writes=count, seconds=time.perf_counter()-t0))
    except Exception:
        output.put(dict(worker=worker, ok=False, error=traceback.format_exc()))


def process_reader(path, ready, start, output, count):
    try:
        ready.put("reader")
        start.wait(10)
        reads = 0
        with MemoryStore(path, principal_id="shared", timeout=15) as store:
            for _ in range(count):
                for f in store.facts():
                    left, right = f["key"].split(".")
                    expected = int(left[6:])*10000+int(right[4:])
                    if f["value"] != expected or not store.get_message(f["source_event_id"]):
                        raise AssertionError("reader observed malformed or sourceless fact")
                store.recall("worker0 worker1 worker2 worker3", limit=8)
                reads += 1
        output.put(dict(worker="reader", ok=True, read_rounds=reads))
    except Exception:
        output.put(dict(worker="reader", ok=False, error=traceback.format_exc()))


def multiprocess(root):
    path = root / "concurrent.sqlite3"
    with MemoryStore(path, principal_id="shared"):
        pass
    ready, output, start = CTX.Queue(), CTX.Queue(), CTX.Event()
    procs = [CTX.Process(target=process_writer, args=(path,w,ready,start,output,100)) for w in range(4)]
    procs.append(CTX.Process(target=process_reader,args=(path,ready,start,output,120)))
    for p in procs:
        p.start()
    for _ in procs:
        ready.get(timeout=20)
    start.set()
    reports = [output.get(timeout=45) for _ in procs]
    for p in procs:
        p.join(5)
        if p.is_alive():
            p.kill(); p.join()
    with MemoryStore(path, principal_id="shared") as store:
        facts, messages = store.facts(), store.messages(limit=1000)
        integrity = store.db.execute("PRAGMA integrity_check").fetchone()[0]
        valid = len(facts) == len(messages) == 400 and integrity == "ok"
        for w in range(4):
            for i in range(100):
                valid &= store.facts(f"worker{w}.item{i}")[0]["value"] == w*10000+i
    return dict(passed=valid and all(r["ok"] for r in reports) and all(p.exitcode == 0 for p in procs),
                expectation="Four OS writers plus one OS reader retain all acknowledged messages and accepted values without malformed reads.",
                worker_reports=reports, process_exitcodes=[p.exitcode for p in procs],
                fact_count=len(facts), message_count=len(messages), sqlite_integrity=integrity)


def process_initializer(path, number, ready, start, output):
    try:
        ready.put(number); start.wait(10)
        with MemoryStore(path, principal_id="init", timeout=10) as store:
            store.append_message("init", "user", f"initialization {number}", event_id=str(number))
        output.put(dict(worker=number, ok=True))
    except Exception:
        output.put(dict(worker=number, ok=False, error=traceback.format_exc()))


def concurrent_initialization(root):
    trials = []
    for trial in range(20):
        path = root / f"init-{trial}.sqlite3"
        ready, output, start = CTX.Queue(), CTX.Queue(), CTX.Event()
        procs = [CTX.Process(target=process_initializer,args=(path,i,ready,start,output)) for i in range(6)]
        for p in procs: p.start()
        for _ in procs: ready.get(timeout=20)
        start.set()
        reports = [output.get(timeout=20) for _ in procs]
        for p in procs: p.join(5)
        with MemoryStore(path,principal_id="init") as store:
            count = len(store.messages())
        trials.append(dict(trial=trial, reports=reports, messages=count,
                           passed=count == 6 and all(r["ok"] for r in reports)))
    return dict(passed=all(x["passed"] for x in trials),
                expectation="Simultaneous initializers can open a new shared database and preserve each acknowledged write.",
                trials=trials)


def dependency_erasure_and_retries(root):
    path = root / "erasure.sqlite3"
    with MemoryStore(path,principal_id="alice") as store:
        store.append_message("root", "user", "rootsecretcanary pebble", event_id="source")
        source = "source"
        for i in range(30):
            event = f"chain-{i}"
            store.append_message(f"chat-{i}","assistant",f"derivedsecretcanary step{i}",event_id=event,source_event_ids=[source])
            source = event
        accepted(store,"derived.claim","sensitive synthetic value",source,"derived-proposal")
        for i in range(5):
            store.append_message("branch","assistant",f"branchsecretcanary {i}",event_id=f"branch-{i}",source_event_ids=["chain-5"])
        session = MemorySession(store,session_id="reply")
        response = session.turn("What is rootsecretcanary?",lambda _q,_ctx:"rootsecretcanary is pebble",request_id="retry-old")
        with MemoryStore(path,principal_id="alice",namespace="other") as foreign:
            foreign.append_message("root","user","foreignsurvivor",event_id="source")
            erased = store.forget_event("source")
            foreign_survives = len(foreign.messages()) == 1
        retry_error = exception_name(lambda: session.turn("What is rootsecretcanary?",lambda _q,_ctx:"must not run",request_id="retry-old"))
        surviving_events = [m["event_id"] for m in store.messages(limit=1000)]
        stale_dependencies = store.db.execute("SELECT COUNT(*) FROM messages WHERE principal='alice' AND namespace='default' AND role='assistant' AND deleted=0").fetchone()[0]
        values = store.facts()
        history = store.history("derived.claim")
    return dict(passed=retry_error == "DeletedError" and foreign_survives and not values and stale_dependencies == 0,
                expectation="Erasing a root transitively removes cross-chat descendants, prevents exact old response retry, and leaves other namespaces intact.",
                erased=erased, old_response_id=response.assistant_event_id, retry_error=retry_error,
                surviving_events=surviving_events, surviving_facts=values, redacted_history=history,
                foreign_namespace_survives=foreign_survives)


def in_flight_worker(path, channel, mutation):
    try:
        with MemoryStore(path,principal_id="alice") as store:
            session=MemorySession(store,session_id="inflight")
            def respond(_message,context):
                channel.send(dict(ready=True,context=context))
                if channel.recv() != "continue": raise RuntimeError("bad barrier")
                return "Orion launches Thursday."
            try:
                session.turn("When does Orion launch?",respond,request_id="inflight-request")
                channel.send(dict(result="committed"))
            except Exception as exc:
                channel.send(dict(result=type(exc).__name__))
    except Exception:
        channel.send(dict(error=traceback.format_exc()))


def inflight_mutations(root):
    results=[]
    for mutation in ("correction","erasure"):
        path=root/f"inflight-{mutation}.sqlite3"
        with MemoryStore(path,principal_id="alice") as store:
            store.append_message("source","user","Orion launches Thursday",event_id="source")
            accepted(store,"orion.launch","Thursday","source","p1")
            parent,child=CTX.Pipe()
            proc=CTX.Process(target=in_flight_worker,args=(path,child,mutation));proc.start()
            if not parent.poll(15): raise TimeoutError("no callback barrier")
            prepared=parent.recv()
            if not prepared.get("ready"): raise RuntimeError(prepared)
            if mutation == "correction":
                store.append_message("correction","user","Orion moved to Friday",event_id="correction")
                accepted(store,"orion.launch","Friday","correction","p2",1)
            else:
                store.forget_event("source")
            parent.send("continue")
            if not parent.poll(15): raise TimeoutError("no commit response")
            outcome=parent.recv();proc.join(5)
            replies=[m for m in store.messages("inflight") if m["role"] == "assistant"]
            expected="ConflictError" if mutation == "correction" else "DeletedError"
            results.append(dict(mutation=mutation,outcome=outcome,assistant_count=len(replies),
                                passed=outcome.get("result") == expected and not replies))
    return dict(passed=all(r["passed"] for r in results),expectation="A second OS process changing recalled evidence before response commit rejects the stale reply.",observations=results)


def crash_writer(path,channel,stage):
    with MemoryStore(path,principal_id="alice") as store:
        if stage == "before_commit":
            def traced(sql):
                if sql == "COMMIT":
                    channel.send("before_commit")
                    channel.recv()
            store.db.set_trace_callback(traced)
        store.append_message("crash","assistant","crashcanary " + "x"*200000,event_id="reply",source_event_ids=["source"])
        if stage == "after_ack":
            channel.send("after_ack")
            channel.recv()


def crash_boundaries(root):
    results=[]
    for stage in ("before_commit","after_ack"):
        path=root/f"crash-{stage}.sqlite3"
        with MemoryStore(path,principal_id="alice") as store:
            store.append_message("seed","user","synthetic source",event_id="source")
        parent,child=CTX.Pipe()
        proc=CTX.Process(target=crash_writer,args=(path,child,stage));proc.start()
        if not parent.poll(15): raise TimeoutError("no crash barrier")
        reached=parent.recv()
        os.kill(proc.pid,signal.SIGKILL);proc.join(5)
        with MemoryStore(path,principal_id="alice") as store:
            present=store.get_message("reply") is not None
            fts_count=len(store.messages(query="crashcanary"))
            edges=store.db.execute("SELECT COUNT(*) FROM message_sources WHERE event='reply'").fetchone()[0]
            integrity=store.db.execute("PRAGMA integrity_check").fetchone()[0]
            valid=present == (stage == "after_ack") and fts_count == int(present) and edges == int(present) and integrity == "ok"
            results.append(dict(stage=stage,reached=reached,exitcode=proc.exitcode,
                                message_present=present,fts_rows=fts_count,dependency_edges=edges,sqlite_integrity=integrity,passed=valid))
    return dict(passed=all(x["passed"] for x in results),expectation="SIGKILL before COMMIT leaves no partial message/index/dependency; SIGKILL after acknowledgment preserves all three.",observations=results)


def checkpoints_and_restore(root):
    path=root/"checkpoint.sqlite3"
    old=root/"old-backup.sqlite3";new=root/"post-erasure-backup.sqlite3"
    manifest={"model":"synthetic-state-only"}
    request_hash=hashlib.sha256(b"inert-input").hexdigest()
    with MemoryStore(path,principal_id="alice") as store:
        store.append_message("source-chat","user","backupcanary synthetic",event_id="source")
        store.commit_checkpoint("decode",request_id="r1",request_hash=request_hash,expected_revision=0,
                                manifest=manifest,payload={"cached":"backupcanary"},result={"answer":"backupcanary"})
        store.backup(old)
        store.forget_event("source")
        invalidated=exception_name(lambda:store.load_checkpoint("decode"))
        retry=store.commit_checkpoint("decode",request_id="r1",request_hash=request_hash,expected_revision=0,
                                      manifest=manifest,payload={"cached":"backupcanary"},result={"answer":"backupcanary"})
        still_invalidated=exception_name(lambda:store.load_checkpoint("decode"))
        store.backup(new)
    new_restore=root/"restored-new.sqlite3";old_restore=root/"restored-old.sqlite3"
    MemoryStore.restore_backup(new,new_restore);MemoryStore.restore_backup(old,old_restore)
    with MemoryStore(new_restore,principal_id="alice") as restored:
        new_source=exception_name(lambda:restored.get_message("source"))
        new_checkpoint=exception_name(lambda:restored.load_checkpoint("decode"))
        receipt=restored.checkpoint_receipt("decode","r1")
    with MemoryStore(old_restore,principal_id="alice") as restored:
        old_source_restored=restored.get_message("source") is not None
        old_checkpoint_restored=restored.load_checkpoint("decode") is not None
    return dict(passed=invalidated == still_invalidated == new_source == new_checkpoint == "DeletedError" and retry["result"] is None and receipt["result"] is None,
                expectation="Erasure removes checkpoint payloads; retries and post-erasure restore cannot silently recover the erased state.",
                invalidated_error=invalidated,retry=retry,post_retry_error=still_invalidated,
                restored_source_error=new_source,restored_checkpoint_error=new_checkpoint,restored_receipt=receipt,
                documented_limit={"pre_erasure_backup_restores_source":old_source_restored,
                                  "pre_erasure_backup_restores_checkpoint":old_checkpoint_restored,
                                  "meaning":"Old backups remain separate retained copies. No automatic propagation of later erasures is implemented."})


def correction_vs_exact_retry(root):
    with MemoryStore(root/"retry.sqlite3",principal_id="alice") as store:
        store.append_message("seed","user","Orion launches Thursday",event_id="source")
        accepted(store,"orion.launch","Thursday","source","p1")
        session=MemorySession(store,session_id="qa")
        calls=[]
        def respond(_message,context):
            calls.append(1)
            return next(f["value"] for f in context["accepted_facts"] if f["key"] == "orion.launch")
        first=session.turn("When does Orion launch?",respond,request_id="r1")
        store.append_message("correct","user","Orion moved to Friday",event_id="correction")
        accepted(store,"orion.launch","Friday","correction","p2",1)
        old_retry=session.turn("When does Orion launch?",respond,request_id="r1")
        fresh=session.turn("When does Orion launch?",respond,request_id="r2")
        return dict(passed=first.response == old_retry.response == "Thursday" and old_retry.replayed and fresh.response == "Friday" and len(calls) == 2,
                    expectation="Exact committed request replay preserves its original answer; a fresh request sees the correction.",
                    original=first.response,old_retry=old_retry.response,old_retry_replayed=old_retry.replayed,fresh_request=fresh.response,
                    callback_calls=len(calls),interpretation="An idempotent retry is not a fresh factual answer; callers must use a new request ID to request a new answer.")


CASES=[ranking_interference,unrelated_facts,multiprocess,concurrent_initialization,
       dependency_erasure_and_retries,inflight_mutations,crash_boundaries,
       checkpoints_and_restore,correction_vs_exact_retry]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results_latest.json"))
    parser.add_argument("--case",action="append",choices=[c.__name__ for c in CASES])
    args=parser.parse_args()
    source={name:hashlib.sha256((ROOT/"nanocog"/"memory"/name).read_bytes()).hexdigest()
            for name in ("store.py","session.py")}
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    report=dict(kind="independent-synthetic-system-experiments",git_head=commit,source_sha256=source,
                python=platform.python_version(),sqlite=sqlite3.sqlite_version,platform=platform.platform(),
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),cases=[])
    for case in CASES:
        if args.case and case.__name__ not in args.case: continue
        t0=time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix=f"cog-validation-{case.__name__}-") as td:
                result=case(Path(td))
                filesystem=subprocess.run(['stat','-f','-c','%T',td],capture_output=True,text=True,check=True).stdout.strip()
                result['temporary_filesystem']=filesystem
            result["status"]="PASS" if result["passed"] else "FAIL"
        except Exception:
            result=dict(passed=False,status="ERROR",error=traceback.format_exc())
        result.update(name=case.__name__,seconds=time.perf_counter()-t0)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps(dict(case=case.__name__,status=result["status"],seconds=round(result["seconds"],3))),flush=True)
    report["counts"]={status:sum(c["status"] == status for c in report["cases"]) for status in ("PASS","FAIL","ERROR")}
    report["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    report['source_sha256_after']={name:hashlib.sha256((ROOT/'nanocog'/'memory'/name).read_bytes()).hexdigest()
                                    for name in source}
    report['source_unchanged']=report['source_sha256_after'] == source
    if not report['source_unchanged']:
        report['counts']['ERROR'] += 1
        report['run_error']='Source files changed during this run; reject mixed-code evidence.'
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report["counts"]),flush=True)
    return 1 if report['counts']['FAIL'] or report['counts']['ERROR'] else 0


if __name__ == "__main__":
    raise SystemExit(main())
