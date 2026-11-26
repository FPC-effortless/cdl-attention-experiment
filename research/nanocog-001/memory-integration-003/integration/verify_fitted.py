#!/usr/bin/env python3
"""Independent real-checkpoint archive interventions and persistence checks.

The accepted-fact panel deliberately supplies a correct ledger to isolate the
learned query/answer connection. It is NOT a learned-writer accuracy metric.
The separate hand-authored history pair uses only raw text at inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "memory-integration-003/learned"))
from runner import load_adapter
from dataset import SYNTAX
from nanocog.memory.learned_archive import (
    ANSWERS, BOXES, ITEMS, UNKNOWN, FactualMemory, fact_key, fact_payload,
    operation, payload_value, execute_operation,
)
from nanocog.memory.store import MemoryStore, ConflictError, DeletedError


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(store, record, value, event):
    source = store.append_message("accepted-source", "tool", f"Controlled fact intervention {event}.",
                                  event_id=event)
    store.propose(fact_key(record), fact_payload(record, value), source["event_id"],
                  proposal_id=event + ":proposal", expected_revision=store.fact_revision(fact_key(record)))
    return store.accept(event + ":proposal", authority="independent-controlled-intervention")


def populate(store):
    for r in range(8):
        put(store, r, r if r < 4 else 8 + r - 4, f"initial-{r}")


def ask(memory, text, request):
    tick = time.perf_counter()
    result = memory.ask(text, request_id=request)
    return {**asdict(result), "wall_ms": 1000 * (time.perf_counter() - tick)}


def causal_panel(adapter, work):
    rows = []
    for syntax, forms in SYNTAX.items():
        for route in range(8):
            path = work / f"causal-{syntax}-{route}.sqlite3"
            with MemoryStore(path, principal_id="causal-panel") as store:
                populate(store)
                memory = FactualMemory(store, adapter, session_id="query-chat", authority="model-policy")
                query = forms["query_box" if route < 4 else "query_item"].format(
                    box=BOXES[route] if route < 4 else BOXES[0],
                    item=ITEMS[route - 4] if route >= 4 else ITEMS[0])
                target = route if route < 4 else route - 4
                first = ask(memory, query, "before")
                put(store, target, target + 4, "relevant-owner")
                relevant = ask(memory, query, "relevant")
                other = (target + 1) % 4
                put(store, other, other + 4, "irrelevant-owner")
                irrelevant = ask(memory, query, "irrelevant")
                pair_ok = (first["answer_index"] == target and relevant["answer_index"] == target + 4)
                unchanged = relevant["answer"] == irrelevant["answer"]
                row = {"syntax": syntax, "route": route, "query": query,
                       "correct_initial_route": first["route"] == route,
                       "relevant_owner_pair_correct": pair_ok,
                       "irrelevant_answer_invariant": unchanged,
                       "irrelevant_readset_invariant": relevant["dependencies"] == irrelevant["dependencies"],
                       "before": first, "relevant": relevant, "irrelevant": irrelevant}
                if route >= 4:
                    put(store, route, 8 + other, "relevant-link")
                    linked = ask(memory, query, "linked")
                    row.update(linked=linked, relevant_link_pair_correct=(
                        irrelevant["answer_index"] == target + 4 and linked["answer_index"] == other + 4))
                rows.append(row)
    groups = {}
    for syntax in SYNTAX:
        sample = [r for r in rows if r["syntax"] == syntax]
        groups[syntax] = {"cases": len(sample),
                          "correct_routes": sum(r["correct_initial_route"] for r in sample),
                          "relevant_owner_pairs": sum(r["relevant_owner_pair_correct"] for r in sample),
                          "irrelevant_invariants": sum(r["irrelevant_answer_invariant"] for r in sample),
                          "irrelevant_readset_invariants": sum(r["irrelevant_readset_invariant"] for r in sample),
                          "relevant_link_pairs": sum(r.get("relevant_link_pair_correct", False) for r in sample),
                          "link_cases": sum("linked" in r for r in sample)}
    return {"scope": "correct accepted ledger supplied; writer excluded", "groups": groups, "cases": rows}


def live_ledger(store):
    return [payload_value(r, facts[0]["value"]) if (facts := store.facts(fact_key(r))) else UNKNOWN
            for r in range(8)]


def independent_history_pair(adapter, work):
    # These texts and labels are hand-authored, not sampled from the experiment's
    # semantic generator. Only each text reaches memory.observe.
    common = [("Amara holds the chest.", operation(0, 0)),
              ("Bela holds the crate.", operation(1, 1)),
              ("Cato holds the basket.", operation(2, 2)),
              ("Dina holds the trunk.", operation(3, 3)),
              ("The coin is in the chest.", operation(4, 8)),
              ("Evan did not put the ring in the basket.", 0)]
    worlds = []
    for name, target_text, target_op, expected in (
            ("transfer", "Amara gave the chest to Fara.", operation(0, 5), "Fara"),
            ("cancel", "Amara did not give the chest to Fara.", 0, "Amara")):
        events = common + [(target_text, target_op), ("Bela gave the crate to Gus.", operation(1, 6))]
        ledger, trace = [UNKNOWN] * 8, []
        with MemoryStore(work / f"manual-{name}.sqlite3", principal_id="manual") as store:
            memory = FactualMemory(store, adapter, session_id="observations", authority="learned-model-policy")
            for i, (text, gold) in enumerate(events):
                actual = memory.observe(text, request_id=f"event-{i}")
                execute_operation(ledger, gold)
                observed = live_ledger(store)
                trace.append({"index": i, "text": text, "expected_operation": gold,
                              "actual": asdict(actual), "correct_operation": actual.operation == gold,
                              "complete_ledger_correct": observed == ledger,
                              "expected_ledger": list(ledger), "actual_ledger": observed})
            restarted = FactualMemory(store, adapter, session_id="new-chat", authority="learned-model-policy")
            answer = ask(restarted, "Who holds the box with the coin?", "question")
            worlds.append({"name": name, "events": trace, "answer": answer,
                           "expected_answer": expected, "answer_correct": answer["answer"] == expected,
                           "all_operations_and_ledgers_correct": all(
                               t["correct_operation"] and t["complete_ledger_correct"] for t in trace)})
    return {"scope": "independent eight-event raw-text pair; not a statistical qualification set",
            "strict_pair_correct": all(w["answer_correct"] and w["all_operations_and_ledgers_correct"] for w in worlds),
            "worlds": worlds}


def guard_panel(adapter, work):
    rows = []
    for scenario in ("correction", "erasure", "absent-insertion"):
        path = work / f"guard-{scenario}.sqlite3"
        with MemoryStore(path, principal_id="guard") as store, MemoryStore(path, principal_id="guard") as other:
            if scenario != "absent-insertion":
                populate(store)
            memory = FactualMemory(store, adapter, session_id="guard-chat", authority="learned-model-policy")
            query = "Who holds the chest?"
            route = int(adapter.predict_queries([query])[0])
            original = adapter.predict_answers
            called = []
            applied = []

            def concurrent_change(values):
                answer = original(values)
                called.append(True)
                previous_revision = other.fact_revision(fact_key(route))
                if scenario == "erasure":
                    fact = other.facts(fact_key(route))[0]
                    other.forget_event(fact["source_event_id"])
                else:
                    value = 7 if route < 4 else 11
                    put(other, route, value, "intervening-write")
                snapshot = other.fact_snapshot(fact_key(route))
                if snapshot["revision"] <= previous_revision:
                    raise AssertionError("the race intervention did not change the fact revision")
                if scenario == "erasure" and snapshot["record"] is not None:
                    raise AssertionError("the erasure intervention did not erase the fact")
                applied.append(snapshot["revision"])
                return answer

            adapter.predict_answers = concurrent_change
            error = None
            try:
                memory.ask(query, request_id="race")
            except (ConflictError, DeletedError) as exc:
                error = type(exc).__name__
            finally:
                adapter.predict_answers = original
            _, response_id, _ = memory._ids("race", "ask")
            persisted = store.get_message(response_id)
            rows.append({"scenario": scenario, "actual_predicted_route": route,
                         "actual_fitted_answer_head_called": len(called) == 1,
                         "intervention_completed": len(applied) == 1,
                         "exception": error, "stale_response_persisted": persisted is not None,
                         "passed": len(called) == 1 and len(applied) == 1 and error is not None and persisted is None})
    return {"scope": "actual fitted prediction with a deterministic concurrent storage mutation",
            "passed": sum(r["passed"] for r in rows), "total": len(rows), "cases": rows}


def restart_check(adapter, work, kit, head):
    path = work / "restart.sqlite3"
    query = "Who holds the chest?"
    with MemoryStore(path, principal_id="restart") as store:
        populate(store)
        memory = FactualMemory(store, adapter, session_id="first-chat", authority="learned-model-policy")
        first = ask(memory, query, "fixed-request")
        put(store, 0, 6, "later-correction")
        replay = ask(memory, query, "fixed-request")
        fresh = ask(memory, query, "fresh-request")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--kit", str(kit),
               "--head", str(head), "--work", str(work), "--output", str(work / "child.json")]
    completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=90)
    child = json.loads((work / "child.json").read_text())
    same_receipt = all(first[k] == replay[k] for k in first if k not in ("wall_ms", "replayed"))
    return {"same_request_replays_original_committed_receipt": same_receipt and replay["replayed"],
            "fresh_process_new_chat_matches_fresh_answer": child["answer"] == fresh["answer"],
            "fresh_process_new_chat_matches_readset": child["dependencies"] == fresh["dependencies"],
            "semantic_original_correct": first["answer"] == "Amara",
            "semantic_corrected_correct": fresh["answer"] == "Gus" and child["answer"] == "Gus",
            "first": first, "replay": replay, "fresh": fresh, "child": child,
            "child_exit_code": completed.returncode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.work.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = torch.load(args.head, map_location="cpu", weights_only=True)
    if checkpoint_metadata.get("arm") != "candidate" or checkpoint_metadata.get("seed") != 17:
        raise ValueError("this audit preregistered candidate seed17; do not substitute a selected checkpoint")
    adapter = load_adapter(args.kit, args.head)
    if isinstance(adapter, tuple):
        adapter = adapter[0]
    adapter.eval()
    if args.child:
        with MemoryStore(args.work / "restart.sqlite3", principal_id="restart") as store:
            memory = FactualMemory(store, adapter, session_id="fresh-process-chat", authority="learned-model-policy")
            result = ask(memory, "Who holds the chest?", "after-process-restart")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if any(args.work.glob("*.sqlite3")):
        raise ValueError("use a fresh work directory to preserve original evidence")
    identity = adapter.model_identity()
    t0, c0 = time.perf_counter(), time.process_time()
    result = {"checkpoint": args.head.name, "checkpoint_sha256": sha(args.head),
              "selection": "prespecified candidate seed17; no best-seed selection",
              "model_identity": identity,
              "causal": causal_panel(adapter, args.work),
              "manual_history_pair": independent_history_pair(adapter, args.work),
              "concurrent_changes": guard_panel(adapter, args.work),
              "restart_and_retry": restart_check(adapter, args.work, args.kit, args.head)}
    result.update(parent_cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0,
                  unchanged_model_identity=adapter.model_identity() == identity,
                  script_sha256=sha(__file__),
                  adapter_source_sha256=sha(ROOT / "nanocog/memory/learned_archive.py"))
    restart = result["restart_and_retry"]
    infrastructure_checks = {
        "all_concurrent_changes_rejected": result["concurrent_changes"]["passed"] == result["concurrent_changes"]["total"],
        "unchanged_model_identity": result["unchanged_model_identity"],
        "same_request_replays_original_committed_receipt": restart["same_request_replays_original_committed_receipt"],
        "fresh_process_new_chat_matches_fresh_answer": restart["fresh_process_new_chat_matches_fresh_answer"],
        "fresh_process_new_chat_matches_readset": restart["fresh_process_new_chat_matches_readset"],
    }
    result["infrastructure_gate"] = {"passed": all(infrastructure_checks.values()), "checks": infrastructure_checks}
    result["semantic_status"] = "Individual findings below; this audit does not replace the whole-history qualification gate."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = {"checkpoint": args.head.name, "causal_groups": result["causal"]["groups"],
               "manual_strict_pair": result["manual_history_pair"]["strict_pair_correct"],
               "guards": {k: v for k, v in result["concurrent_changes"].items() if k != "cases"},
               "restart": {k: v for k, v in result["restart_and_retry"].items() if isinstance(v, (bool, int))},
               "unchanged_model_identity": result["unchanged_model_identity"]}
    summary["infrastructure_gate"] = result["infrastructure_gate"]
    print(json.dumps(summary, indent=2))
    if not result["infrastructure_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
