#!/usr/bin/env python3
"""Post-fit diagnostic replay of frozen predictions; never fits or calls a model.

Gold routes, operations and ledgers are used only by this independent scorer.
There is no oracle intervention and no change to the primary qualification gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LEARNED = HERE.parent / "learned"
UNKNOWN = 12
SEEDS = (17, 29, 41)
ARMS = ("control", "candidate")
PANELS = ("control_train", "dev", "familiar_test", "fresh_test")
CATEGORIES = ("correct_noop", "correct_write", "missed_write", "unrequested_write",
              "wrong_address_only", "wrong_value_only", "wrong_address_and_value")
NATIVE_SHA = "d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def decode(index):
    assert type(index) is int and 0 <= index <= 96, ("invalid operation", index)
    return None if index == 0 else divmod(index - 1, 12)


def valid(proposal):
    if proposal is None:
        return False
    record, value = proposal
    return (0 <= record < 4 and 0 <= value < 8) or (4 <= record < 8 and 8 <= value < 12)


def apply(ledger, operation):
    proposal = decode(operation)
    if valid(proposal):
        ledger[proposal[0]] = proposal[1]


def read(ledger, route):
    assert type(route) is int and 0 <= route < 8
    value = ledger[route]
    if route >= 4:
        return ledger[value - 8] if 8 <= value < 12 else UNKNOWN
    return value if 0 <= value < 8 else UNKNOWN


def error_category(required, predicted):
    gold, proposal = decode(required), decode(predicted)
    assert gold is None or valid(gold), "invalid generator operation"
    if gold is None:
        return "correct_noop" if proposal is None else "unrequested_write"
    if proposal is None:
        return "missed_write"
    address_error, value_error = proposal[0] != gold[0], proposal[1] != gold[1]
    if address_error and value_error:
        return "wrong_address_and_value"
    if address_error:
        return "wrong_address_only"
    if value_error:
        return "wrong_value_only"
    return "correct_write"


def world_row(world, predicted, world_index):
    ledger, previous_gold = [UNKNOWN] * 8, [UNKNOWN] * 8
    categories, attempted, effective = Counter(), Counter(), Counter()
    wrong_operations = state_errors = invalid_count = damage_count = 0
    first_error = first_bad_ledger = first_damage = None
    for position, event in enumerate(world["events"]):
        required, chosen = event["operation"], predicted[event["text"]][0]
        category = error_category(required, chosen)
        categories[category] += 1
        proposal = decode(chosen)
        invalid = proposal is not None and not valid(proposal)
        invalid_count += invalid
        expected_commit = int(required != 0)
        attempted[f"expected{expected_commit}_predicted{int(chosen != 0)}"] += 1
        effective[f"expected{expected_commit}_predicted{int(valid(proposal))}"] += 1
        # Independently verify supplied gold ledgers, rather than silently trust
        # a malformed scoring payload or import the training evaluator.
        independently_updated_gold = list(previous_gold)
        apply(independently_updated_gold, required)
        assert independently_updated_gold == event["ledger"], "generator ledger mismatch"
        before = list(ledger)
        apply(ledger, chosen)
        if chosen != required:
            wrong_operations += 1
            if first_error is None:
                first_error = {"world": world_index, "event": position, "predicted": chosen,
                               "required": required, "category": category, "invalid": invalid}
        if ledger != event["ledger"]:
            state_errors += 1
            if first_bad_ledger is None:
                first_bad_ledger = position
        damaged = [r for r in range(8) if before[r] == previous_gold[r] != UNKNOWN
                   and previous_gold[r] == event["ledger"][r] and ledger[r] != event["ledger"][r]]
        damage_count += len(damaged)
        if damaged and first_damage is None:
            first_damage = {"world": world_index, "event": position, "records": damaged,
                            "category": category, "predicted": chosen, "required": required}
        previous_gold = event["ledger"]
    gold_route, predicted_route = world["route"], predicted[world["query"]][1]
    required_records = [gold_route]
    if gold_route >= 4:
        assert 8 <= previous_gold[gold_route] < 12
        required_records.append(previous_gold[gold_route] - 8)
    chain_correct = all(ledger[r] == previous_gold[r] for r in required_records)
    assert read(previous_gold, gold_route) == world["answer"], "generator answer mismatch"
    value = read(ledger, predicted_route)
    answer = predicted["__answer_dictionary__"][value]
    return {
        "world": world_index, "operations": len(world["events"]),
        "operation_categories": {k: categories[k] for k in CATEGORIES},
        "attempted_commit_confusion": dict(attempted), "effective_commit_confusion": dict(effective),
        "wrong_operations": wrong_operations, "post_event_ledger_errors": state_errors,
        "invalid_operations": invalid_count, "damaged_previously_correct_facts": damage_count,
        "first_error": first_error, "first_bad_ledger_event": first_bad_ledger,
        "first_damage": first_damage, "final_ledger": ledger,
        "final_ledger_correct": ledger == previous_gold,
        "required_chain_records": required_records,
        "required_chain_correct": chain_correct,
        "required_chain_wrong_records": [r for r in required_records if ledger[r] != previous_gold[r]],
        "predicted_route": predicted_route, "route_correct": predicted_route == gold_route,
        "read_value": value, "answer": answer, "answer_correct": answer == world["answer"],
    }


def pair_row(pair, predicted):
    worlds = [world_row(world, predicted, index) for index, world in enumerate(pair["worlds"])]
    assert len(worlds) == 2
    row = {key: pair[key] for key in ("lineage", "index", "length", "syntax")}
    row.update(worlds=worlds,
               all_operations=all(w["wrong_operations"] == 0 for w in worlds),
               all_post_event_ledgers=all(w["post_event_ledger_errors"] == 0 for w in worlds),
               both_final_ledgers=all(w["final_ledger_correct"] for w in worlds),
               both_answers=all(w["answer_correct"] for w in worlds),
               both_query_routes=all(w["route_correct"] for w in worlds),
               both_required_chains=all(w["required_chain_correct"] for w in worlds))
    row["strict_pair"] = row["all_operations"] and row["all_post_event_ledgers"] and row["both_answers"]
    row["both_required_chains_and_answers"] = row["both_required_chains"] and row["both_answers"]
    row["both_required_chains_routes_answers"] = (row["both_required_chains"] and
                                                  row["both_query_routes"] and row["both_answers"])
    for key in ("wrong_operations", "post_event_ledger_errors", "invalid_operations",
                "damaged_previously_correct_facts"):
        row[key] = sum(w[key] for w in worlds)
    row["operations_count"] = sum(w["operations"] for w in worlds)
    row["answers"] = [w["answer"] for w in worlds]
    row["routes"] = [w["predicted_route"] for w in worlds]
    # The locked runner scans world0 then world1. Preserve that definition only
    # for reconciliation, and additionally report per-world and earliest-event
    # diagnostics so event position is not confused with world traversal order.
    first = next((w["first_error"] for w in worlds if w["first_error"] is not None), None)
    row["first_error"] = ({k: first[k] for k in ("world", "event", "predicted", "required")}
                          if first is not None else None)
    row["earliest_pair_error_event"] = min((w["first_error"]["event"] for w in worlds
                                             if w["first_error"] is not None), default=None)
    return row


def aggregate(rows):
    assert rows
    worlds = [w for row in rows for w in row["worlds"]]
    categories, attempted, effective = Counter(), Counter(), Counter()
    for world in worlds:
        categories.update(world["operation_categories"])
        attempted.update(world["attempted_commit_confusion"])
        effective.update(world["effective_commit_confusion"])
    result = {"pairs": len(rows), "histories": len(worlds),
              "operations": sum(r["operations_count"] for r in rows),
              "operation_categories": {k: categories[k] for k in CATEGORIES},
              "attempted_commit_confusion": dict(attempted), "effective_commit_confusion": dict(effective)}
    for metric in ("all_operations", "all_post_event_ledgers", "both_final_ledgers", "both_answers",
                   "both_query_routes", "both_required_chains", "both_required_chains_and_answers",
                   "both_required_chains_routes_answers", "strict_pair"):
        correct = sum(row[metric] for row in rows)
        result[metric] = {"correct": correct, "total": len(rows), "fraction": correct / len(rows)}
    for metric in ("wrong_operations", "post_event_ledger_errors", "invalid_operations",
                   "damaged_previously_correct_facts"):
        result[metric] = sum(r[metric] for r in rows)
    result["both_operations_request_write"] = sum(categories[k] for k in (
        "correct_write", "wrong_address_only", "wrong_value_only", "wrong_address_and_value"))
    result["attempted_commit_errors"] = categories["missed_write"] + categories["unrequested_write"]
    result["effective_commit_errors"] = effective["expected1_predicted0"] + effective["expected0_predicted1"]
    result["required_chain_histories_correct"] = sum(w["required_chain_correct"] for w in worlds)
    result["correct_chain_but_wrong_answer_histories"] = sum(w["required_chain_correct"] and
                                                             not w["answer_correct"] for w in worlds)
    result["wrong_chain_but_correct_answer_histories"] = sum(not w["required_chain_correct"] and
                                                             w["answer_correct"] for w in worlds)
    result["correct_chain_but_wrong_route_histories"] = sum(w["required_chain_correct"] and
                                                            not w["route_correct"] for w in worlds)
    result["histories_with_damage"] = sum(w["first_damage"] is not None for w in worlds)
    for key, source, field in (("first_error_event_counts", "first_error", "event"),
                               ("first_error_category_counts", "first_error", "category"),
                               ("first_damage_event_counts", "first_damage", "event"),
                               ("first_damage_category_counts", "first_damage", "category")):
        result[key] = dict(sorted(Counter(str(w[source][field]) for w in worlds
                                         if w[source] is not None).items()))
    result["first_bad_ledger_event_counts"] = dict(Counter(str(w["first_bad_ledger_event"])
        for w in worlds if w["first_bad_ledger_event"] is not None))
    result["earliest_pair_error_event_counts"] = dict(Counter(str(r["earliest_pair_error_event"])
        for r in rows if r["earliest_pair_error_event"] is not None))
    assert sum(categories.values()) == result["operations"]
    assert result["wrong_operations"] == result["operations"] - categories["correct_noop"] - categories["correct_write"]
    return result


def build_report(result):
    lines = ["# Frozen-prediction diagnostic replay", "", result["scope"], "",
             "The original primary metrics reconcile exactly for every scored pair. "
             "This replay does not change qualification thresholds, fit another model, "
             "select a checkpoint, or apply an oracle repair.", "", "## Definitions", "",
             "A missed write predicts NOOP for a required write; an unrequested write proposes a write "
             "for a gold NOOP. Among cases where both operations request a write, address-only, "
             "value-only and joint address/value failures are mutually exclusive. These are components "
             "of the jointly chosen operation, not separate learned heads. Invalid typed proposals "
             "are reported separately and are rejected by execution, never corrected.", "",
             "Attempted commitment means a non-NOOP prediction. Effective commitment additionally "
             "requires a valid record/value type. The JSON reports both confusion tables; an invalid "
             "write can be an attempted commitment without changing the ledger.", "",
             "Required-chain correctness uses the gold question route and final gold link solely "
             "inside this scorer: one owner record for a box query, or the item link plus its required "
             "owner record for an item query. Every record in that chain must equal the gold final "
             "ledger. Predicted query routes and actual learned answers are assessed separately. "
             "No gold record, route or answer is supplied to the model.", "",
             "Damage means a previously correct, known record whose gold value did not change at "
             "this event became incorrect. First-error and first-damage positions are zero-based and "
             "reported per history; the earliest-event index across a pair is also retained. "
             "This avoids treating world0-first traversal as temporal ordering between worlds.", ""]
    for panel, heading in (("familiar_test", "Familiar TEST"), ("fresh_test", "Fresh TEST")):
        lines += ["## " + heading, "", "| Arm | Seed | Strict pairs | Both required chains | Chains + answers | Both routes | Both answers |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        selected = [run for run in result["runs"] if run["panel"] == panel]
        for run in selected:
            s = run["summary"]; n = s["pairs"]
            counts = [f"{s[k]['correct']}/{n}" for k in (
                "strict_pair", "both_required_chains", "both_required_chains_and_answers",
                "both_query_routes", "both_answers")]
            lines.append(f"| {run['arm']} | {run['seed']} | " + " | ".join(counts) + " |")
        lines += ["", "| Arm | Seed | Missed | Unrequested | Address only | Value only | Address + value | Invalid | Damaged records |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for run in selected:
            s = run["summary"]; c = s["operation_categories"]
            counts = [str(c[k]) for k in ("missed_write", "unrequested_write", "wrong_address_only",
                                         "wrong_value_only", "wrong_address_and_value")]
            counts += [str(s["invalid_operations"]), str(s["damaged_previously_correct_facts"])]
            lines.append(f"| {run['arm']} | {run['seed']} | " + " | ".join(counts) + " |")
        lines += ["", "Counts are event-level; invalid proposals overlap the mutually exclusive "
                  "operation-error categories. The JSON supplies the event and conditional-write "
                  "denominators, first-position histograms, and length breakdowns.", ""]
    lines += ["## Where fresh-history errors begin", "",
              "| Arm | Seed | First event already wrong | Histories with later damage |",
              "|---|---:|---:|---:|"]
    for run in result["runs"]:
        if run["panel"] == "fresh_test":
            s = run["summary"]; n = s["histories"]
            lines.append(f"| {run['arm']} | {run['seed']} | "
                         f"{s['first_error_event_counts'].get('0', 0)}/{n} | {s['histories_with_damage']}/{n} |")
    lines += ["", "These errors include the initial fact-establishment steps. They cannot be "
              "attributed solely to long-history accumulation. Damage counts depend on whether a "
              "history first established a correct record, so raw damage totals are not "
              "exposure-matched error rates.", "", "## Held-out grammar breakdown", "",
              "| Arm | Seed | Grammar | Strict pairs | Both required chains | Both answers |",
              "|---|---:|---|---:|---:|---:|"]
    for run in result["runs"]:
        if run["panel"] != "fresh_test":
            continue
        for grammar, s in run["by_syntax"].items():
            n = s["pairs"]
            lines.append(f"| {run['arm']} | {run['seed']} | {grammar} | " + " | ".join(
                f"{s[k]['correct']}/{n}" for k in ("strict_pair", "both_required_chains", "both_answers")) + " |")
    lines += ["", "## Evidence and limits", "",
              f"Reconciled **{result['reconciliation']['pairs_checked']} pair rows** across all six frozen checkpoints "
              "and the four original panels. Every original field was compared, including first error, "
              "operation count, final ledgers, answers, routes, invalid writes and damage.", "",
              "The three seeds reuse the same test families; they are head-initialization replications "
              "on one frozen native backbone, not independent language models or independent populations. "
              "Counts describe these fixed synthetic histories. A component diagnosis is not evidence "
              "that a proposed repair would succeed. There are no new model calls or counterfactual "
              "oracle corrections in this diagnostic.", "",
              "`diagnostic-summary.json` contains all per-arm/seed/panel/grammar/length counts and "
              "input hashes. Detailed history rows are kept with the private experimental artifacts. "
              "The corpus hash, checkpoint hashes, feature-cache receipt and prediction-export "
              "metadata are validated before replay.", ""]
    return "\n".join(lines)


def main(args):
    wall, cpu = time.perf_counter(), time.process_time()
    script_hash = file_sha(__file__)
    manifest = load(LEARNED / "DATA_MANIFEST.json")
    corpus_path = args.private / "corpus.json"
    assert file_sha(corpus_path) == manifest["corpus_sha256"]
    data = load(corpus_path)["data"]
    original_rows = load(args.private / "PAIR_ROWS.json")
    fits = load(args.private / "FITS.json")
    cache_receipt = load(args.private / "all_features.json")
    cache_hash = file_sha(args.private / "all_features.pt")
    assert cache_hash == cache_receipt["sha256"]
    known_fits = {(row["arm"], row["seed"]): row for row in fits["runs"]}
    assert set(known_fits) == {(a, s) for a in ARMS for s in SEEDS}
    result = {"scope": "Independent, post-fit, pure-Python replay of fixed predictions. "
              "No model calls, additional training, checkpoint selection or oracle corrections.",
              "runs": [], "input_hashes": {"corpus": file_sha(corpus_path),
                  "original_pair_rows": file_sha(args.private / "PAIR_ROWS.json"),
                  "fits": file_sha(args.private / "FITS.json"), "feature_cache": cache_hash,
                  "feature_receipt": file_sha(args.private / "all_features.json")},
              "prediction_hashes": {}, "checkpoint_hashes": {},
              "reconciliation": {"pairs_checked": 0, "mismatches": 0}}
    details = {}
    for seed in SEEDS:
        for arm in ARMS:
            label = f"{arm}-{seed}"
            checkpoint = args.private / "checkpoints" / (label + ".pt")
            checkpoint_hash = file_sha(checkpoint)
            assert checkpoint_hash == known_fits[arm, seed]["file_sha256"]
            export_path = args.private / "predictions" / (label + ".json")
            export = load(export_path)
            assert (export["arm"], export["seed"]) == (arm, seed)
            assert export["checkpoint_sha256"] == checkpoint_hash
            assert export["native_sha256"] == NATIVE_SHA
            assert export["cache_sha256"] == cache_hash
            predicted = export["predictions"]
            dictionary = predicted["__answer_dictionary__"]
            assert len(dictionary) == 13 and all(type(v) is int and 0 <= v < 9 for v in dictionary)
            result["prediction_hashes"][label] = file_sha(export_path)
            result["checkpoint_hashes"][label] = checkpoint_hash
            for panel in PANELS:
                rows = [pair_row(pair, predicted) for pair in data[panel]]
                previous = original_rows[f"{label}-{panel}"]
                assert len(rows) == len(previous)
                for row, old in zip(rows, previous):
                    assert {key: row[key] for key in old} == old, (label, panel, row["index"])
                result["reconciliation"]["pairs_checked"] += len(rows)
                details[f"{label}-{panel}"] = rows
                result["runs"].append({"arm": arm, "seed": seed, "panel": panel,
                    "summary": aggregate(rows),
                    "by_syntax": {syntax: aggregate([r for r in rows if r["syntax"] == syntax])
                                  for syntax in sorted({r["syntax"] for r in rows})},
                    "by_length": {str(length): aggregate([r for r in rows if r["length"] == length])
                                  for length in (8, 24, 64)}})
            if time.process_time() - cpu > 60 or time.perf_counter() - wall > 180:
                raise RuntimeError("diagnostic replay resource bound exceeded")
    assert result["reconciliation"]["pairs_checked"] == 6912
    assert file_sha(__file__) == script_hash
    result["source_sha256"] = script_hash
    result["passed"] = True
    save(args.private / "DIAGNOSTIC_ROWS.json", details)
    args.report.write_text(build_report(result))
    result["cpu_seconds"] = time.process_time() - cpu
    result["wall_seconds"] = time.perf_counter() - wall
    result["timing_scope"] = "validation, replay, private details and report output; excludes final summary write"
    save(args.output, result)
    print(json.dumps({"passed": True, "pairs_reconciled": 6912,
                      "cpu_seconds": result["cpu_seconds"], "wall_seconds": result["wall_seconds"],
                      "output": str(args.output), "report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "diagnostic-summary.json")
    parser.add_argument("--report", type=Path, default=HERE / "DIAGNOSTICS.md")
    main(parser.parse_args())
