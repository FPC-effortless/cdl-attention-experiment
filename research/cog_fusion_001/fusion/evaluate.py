"""Full generated-answer evaluation with per-world causal-path evidence.

Reference labels are inspected only after candidate operations and generation,
except in the explicitly named supplied-reference-evidence diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .data import UNKNOWN
from .engine import (Resolution, execute, greedy_generate, hybrid_read,
                     predicted_operation, strict_read)
from .model import FusionModel, clone_state, state_bytes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operation_equal(predicted, expected):
    op, e, r, v = expected
    return predicted[0] == op and (op == 0 or (predicted[1:3] == (e, r) and (op != 1 or predicted[3] == v)))


def summary(rows):
    modes = sorted({mode for row in rows for mode in row["answers"]})
    result = {"worlds": len(rows), "families": len({row["family_id"] for row in rows}),
              "operations_exact": sum(r["operations_exact"] for r in rows),
              "operations_total": sum(r["operations_total"] for r in rows),
              "all_operations_exact": sum(r["all_operations_exact"] for r in rows),
              "all_post_event_ledgers_exact": sum(r["all_post_event_ledgers_exact"] for r in rows),
              "query_exact": sum(r["query_exact"] for r in rows), "modes": {}}
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    for mode in modes:
        output = {"exact_answers": sum(r["answers"][mode]["correct"] for r in rows),
                  "complete_pairs": 0, "strict_whole_history_pairs": 0,
                  "answer_changing_complete": 0, "answer_changing_total": 0,
                  "invariant_complete": 0, "invariant_total": 0,
                  "support_exact": sum(r["answers"][mode]["support_exact"] for r in rows),
                  "max_candidates_per_query": max((r["answers"][mode]["candidates_fetched"] for r in rows), default=0),
                  "by_kind": {}}
        for pair in families.values():
            if len(pair) != 2:
                raise ValueError("evaluation lost one member of a paired family")
            changing = pair[0]["target"] != pair[1]["target"]
            pair_ok = all(r["answers"][mode]["correct"] for r in pair)
            whole = pair_ok and all(r["all_operations_exact"] and r["all_post_event_ledgers_exact"]
                                   and r["query_exact"] and r["answers"][mode]["support_exact"] for r in pair)
            output["complete_pairs"] += int(pair_ok)
            output["strict_whole_history_pairs"] += int(whole)
            group = "answer_changing" if changing else "invariant"
            output[group + "_complete"] += int(pair_ok)
            output[group + "_total"] += 1
            kind = pair[0]["kind"]
            stat = output["by_kind"].setdefault(kind, {"pairs": 0, "complete_pairs": 0, "strict_pairs": 0, "exact_answers": 0})
            stat["pairs"] += 1
            stat["complete_pairs"] += int(pair_ok)
            stat["strict_pairs"] += int(whole)
            stat["exact_answers"] += sum(r["answers"][mode]["correct"] for r in pair)
        result["modes"][mode] = output
    return result


@torch.inference_mode()
def evaluate_panel(model, families, *, batch_families=8, diagnostics=True):
    rows = []
    started = time.perf_counter()
    for start in range(0, len(families), batch_families):
        batch = families[start:start + batch_families]
        worlds = [world for family in batch for world in family["worlds"]]
        state = model.init_state(len(worlds))
        ledgers = [{} for _ in worlds]
        histories = [[] for _ in worlds]
        all_ledgers = [True] * len(worlds)
        op_correct = [0] * len(worlds)
        op_total = len(worlds[0]["events"])
        if any(len(w["events"]) != op_total for w in worlds):
            raise ValueError("batch must contain equal event counts")
        for position in range(op_total):
            tokens, lengths = model.encode_texts([w["events"][position]["text"] + "\n" for w in worlds])
            _, semantics, state = model.encode_event(tokens, lengths, state)
            for i in range(len(worlds)):
                predicted = predicted_operation(semantics, i)
                execute(ledgers[i], predicted)
                histories[i].append(list(predicted))
                event = worlds[i]["events"][position]
                expected = tuple(event[k] for k in ("operation", "entity", "relation", "value"))
                op_correct[i] += int(operation_equal(predicted, expected))
                actual = [[e, r, v] for (e, r), v in sorted(ledgers[i].items())]
                all_ledgers[i] &= actual == worlds[i]["post_event_ledgers"][position]
        tokens, lengths = model.encode_texts([w["query"] + "\n" for w in worlds])
        features, semantics, state = model.encode_event(tokens, lengths, state)
        entities = semantics["query_entity"].argmax(-1).tolist()
        plans = semantics["query_plan"].argmax(-1).tolist()
        strict = [strict_read(ledger, entity, plan) for ledger, entity, plan in zip(ledgers, entities, plans)]
        modes = {"strict": (strict, None)}
        if diagnostics and model.memory_enabled:
            modes["acute_memory_off"] = (strict, False)
            modes["hybrid"] = ([hybrid_read(model, features[i:i + 1], ledger, plan)
                                for i, (ledger, plan) in enumerate(zip(ledgers, plans))], None)
            # Explicit supervised bypass. It never substitutes into strict/hybrid.
            supplied = [Resolution(list(w["evidence_values"]), [tuple(k) for k in w["supports"]],
                                   "reference", 0) for w in worlds]
            modes["supplied_reference_evidence"] = (supplied, None)
        answers = {}
        for mode, (resolutions, enabled) in modes.items():
            texts, _ = greedy_generate(model, clone_state(state), resolutions, memory_enabled=enabled)
            answers[mode] = (texts, resolutions)
        for i, world in enumerate(worlds):
            fam = batch[i // 2]
            row = {"family_id": fam["family_id"], "side": i % 2, "kind": fam["kind"],
                   "query": world["query"], "target": world["answer_text"],
                   "predicted_query_entity": entities[i], "predicted_query_plan": plans[i],
                   "query_exact": entities[i] == world["query_entity"] and plans[i] == world["query_plan"],
                   "operations_exact": op_correct[i], "operations_total": op_total,
                   "all_operations_exact": op_correct[i] == op_total,
                   "all_post_event_ledgers_exact": bool(all_ledgers[i]),
                   "predicted_operations": histories[i],
                   "final_ledger": [[e, r, v] for (e, r), v in sorted(ledgers[i].items())],
                   "answers": {}}
            for mode, (texts, resolutions) in answers.items():
                res = resolutions[i]
                row["answers"][mode] = {"generated": texts[i], "correct": texts[i] == world["answer_text"],
                                          "support_exact": [list(k) for k in res.keys] == world["supports"],
                                          "keys": res.keys, "values": res.values, "status": res.status,
                                          "candidates_fetched": res.candidates_fetched,
                                          "codebook_comparisons": res.codebook_comparisons,
                                          "index_records": res.index_records}
            rows.append(row)
    return {"summary": summary(rows), "rows": rows, "evaluation_wall_seconds": time.perf_counter() - started,
            "timing_status": "evaluation duration, not isolated serving benchmark"}


@torch.inference_mode()
def fixed_state_interventions(model, families):
    """Control archive content while preserving identical stale neural state.

    Labels define explicit intervention fixtures, not candidate observations.
    These supplied-state tests isolate causal output sensitivity and are not
    substituted for learned writer/query qualification.
    """
    selected = [fam for fam in families if fam["kind"] in ("correction", "bridge", "irrelevant", "cancelled")]
    result = []
    for fam in selected:
        origins = []
        for world in fam["worlds"]:
            state = model.init_state(1)
            for event in world["events"]:
                tokens, lengths = model.encode_texts([event["text"] + "\n"])
                _, _, state = model.encode_event(tokens, lengths, state)
            tokens, lengths = model.encode_texts([world["query"] + "\n"])
            _, _, state = model.encode_event(tokens, lengths, state)
            generated = []
            for variant in fam["worlds"]:
                packet = Resolution(list(variant["evidence_values"]), [tuple(k) for k in variant["supports"]], "reference")
                text, _ = greedy_generate(model, clone_state(state), [packet])
                generated.append(text[0])
            origins.append(generated)
        targets = [w["answer_text"] for w in fam["worlds"]]
        result.append({"family_id": fam["family_id"], "kind": fam["kind"], "generated_by_state_origin": origins,
                       "targets": targets, "both_correct": all(generated == targets for generated in origins),
                       "directional_pairs_correct": sum(generated == targets for generated in origins),
                       "answer_changing": targets[0] != targets[1]})
    groups = {}
    for changing in (True, False):
        subset = [r for r in result if r["answer_changing"] == changing]
        groups["relevant" if changing else "invariant"] = {
            "pairs": len(subset), "both_correct": sum(r["both_correct"] for r in subset),
            "directional_pairs": 2 * len(subset), "directional_pairs_correct": sum(r["directional_pairs_correct"] for r in subset),
            "outputs_changed": sum(any(pair[0] != pair[1] for pair in r["generated_by_state_origin"]) for r in subset)}
    return {"qualification": "supplied evidence interventions in both directions; invariant identical-packet cases test packet stability only",
            "summary": groups, "rows": result}


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    corpus = json.loads(args.corpus.read_text())
    model = FusionModel.from_checkpoint(args.checkpoint).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "RESULT.json").exists():
        raise FileExistsError("evaluation already exists")
    result = {"checkpoint_sha256": sha(args.checkpoint), "corpus_sha256": sha(args.corpus),
              "model_identity": model.model_identity(), "parameters": model.parameter_summary(),
              "state_tensor_bytes": state_bytes(model.init_state(1)), "panels": {}}
    for panel in args.panels:
        observed = evaluate_panel(model, corpus[panel], diagnostics=panel != "long")
        path = args.output / (panel + ".json")
        path.write_text(json.dumps(observed, indent=2) + "\n")
        result["panels"][panel] = observed["summary"]
        print(json.dumps({"panel": panel, "summary": observed["summary"]}), flush=True)
    if model.memory_enabled and "test" in args.panels:
        intervention = fixed_state_interventions(model, corpus["test"])
        (args.output / "interventions.json").write_text(json.dumps(intervention, indent=2) + "\n")
        result["interventions"] = intervention["summary"]
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["test", "fresh", "long"])
    run(parser.parse_args())
