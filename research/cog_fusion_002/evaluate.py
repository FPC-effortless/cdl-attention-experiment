"""Text-only continuous multi-turn rollout and post-inference scoring.

The candidate receives only explicit observe/ask boundaries and text. It never
receives typed writes, query identities, supports, target lengths or answer
strings. Actual generated characters, including the terminator, enter the state
used by later observations. Gold labels are consulted by a separate scorer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from common import PARENT, HERE, sha256, write_json
import torch
from fusion.engine import execute, greedy_generate, hybrid_read, predicted_operation, strict_read
from fusion.model import FusionModel, state_bytes
from data import digest, ledger_rows


COUNT_FIELDS = (
    "operations_exact", "operations_total", "post_answer_operations_exact", "post_answer_operations_total",
    "meaningful_post_answer_operations_exact", "meaningful_post_answer_operations_total",
    "immediate_post_answer_operations_exact", "immediate_post_answer_operations_total",
    "meaningful_immediate_post_answer_operations_exact", "meaningful_immediate_post_answer_operations_total",
    "queries_exact", "queries_total", "support_keys_exact", "support_exact", "query_support_joint_exact",
    "answers_exact", "answers_total", "input_characters", "generated_characters", "control_characters",
    "historical_characters_replayed", "answer_boundaries_reset",
)


def operation_exact(predicted, expected):
    return (predicted[0] == expected[0] and (expected[0] == 0 or
            (tuple(predicted[1:3]) == tuple(expected[1:3]) and
             (expected[0] != 1 or predicted[3] == expected[3]))))


def candidate_inputs(world):
    return [{"type": step["type"], "text": step["text"]} for step in world["steps"]]


@torch.inference_mode()
def rollout_batch(model, input_worlds, *, reset_after_answer=False, memory_enabled=None, reader="strict"):
    """Candidate execution: this function rejects any supplied label fields."""
    if not input_worlds or reader not in ("strict", "hybrid"):
        raise ValueError("provide a nonempty batch and a declared reader")
    schedule = [step["type"] for step in input_worlds[0]]
    for world in input_worlds:
        if [step["type"] for step in world] != schedule:
            raise ValueError("batched worlds require identical boundary schedules")
        if any(set(step) != {"type", "text"} for step in world):
            raise ValueError("candidate rollout accepts only text and explicit boundaries")
    state = model.init_state(len(input_worlds))
    ledgers = [{} for _ in input_worlds]
    traces = [[] for _ in input_worlds]
    for position, kind in enumerate(schedule):
        texts = [world[position]["text"] + "\n" for world in input_worlds]
        tokens, lengths = model.encode_texts(texts)
        features, semantics, state = model.encode_event(tokens, lengths, state)
        if kind == "observe":
            for index in range(len(input_worlds)):
                operation = predicted_operation(semantics, index)
                execute(ledgers[index], operation)
                traces[index].append({"position": position, "type": kind,
                                      "text": input_worlds[index][position]["text"],
                                      "predicted_operation": list(operation),
                                      "post_event_ledger": ledger_rows(ledgers[index]),
                                      "input_characters": int(lengths[index])})
            continue
        if kind != "ask":
            raise ValueError("unknown explicit interaction boundary")
        entities = semantics["query_entity"].argmax(-1).tolist()
        plans = semantics["query_plan"].argmax(-1).tolist()
        if reader == "strict":
            resolutions = [strict_read(ledger, entity, plan) for ledger, entity, plan in zip(ledgers, entities, plans)]
        else:
            resolutions = [hybrid_read(model, features[index:index+1], ledger, plan)
                           for index, (ledger, plan) in enumerate(zip(ledgers, plans))]
        generated, state = greedy_generate(model, state, resolutions, memory_enabled=memory_enabled)
        for index, (text, resolution) in enumerate(zip(generated, resolutions)):
            traces[index].append({"position": position, "type": kind,
                                  "text": input_worlds[index][position]["text"],
                                  "predicted_query_entity": entities[index], "predicted_query_plan": plans[index],
                                  "selected_keys": [list(key) for key in resolution.keys],
                                  "selected_values": list(resolution.values), "evidence_status": resolution.status,
                                  "generated": text, "candidates_fetched": resolution.candidates_fetched,
                                  "codebook_comparisons": resolution.codebook_comparisons,
                                  "index_records": resolution.index_records,
                                  "current_ledger": ledger_rows(ledgers[index]),
                                  "input_characters": int(lengths[index]),
                                  "generated_characters": len(text), "control_characters": len("\nAnswer: "),
                                  "state_reset_after_answer": bool(reset_after_answer)})
        if reset_after_answer:
            # Reset control retains the factual ledger but drops neural tensors.
            state = model.init_state(len(input_worlds))
    return {"traces": traces, "final_state": state, "final_ledgers": [ledger_rows(ledger) for ledger in ledgers]}


def score_world(reference, trace, *, family_id, side, kind, wording):
    """Score a completed candidate trace; never call the model from this path."""
    if len(trace) != len(reference["steps"]):
        raise ValueError("candidate trace omitted transcript steps")
    result = {"family_id": family_id, "side": side, "kind": kind, "wording": wording,
              **{key: 0 for key in COUNT_FIELDS}, "all_post_event_ledgers_exact": True, "steps": []}
    seen_answer = False
    previous_type = None
    turn = 0
    for truth, observed in zip(reference["steps"], trace):
        if truth["type"] != observed["type"] or truth["text"] != observed["text"]:
            raise ValueError("candidate trace inputs differ from the fixed transcript")
        row = dict(observed)
        result["input_characters"] += observed["input_characters"]
        if truth["type"] == "observe":
            expected = [truth[key] for key in ("operation", "entity", "relation", "value")]
            exact = operation_exact(observed["predicted_operation"], expected)
            ledger_exact = observed["post_event_ledger"] == truth["post_event_ledger"]
            meaningful = truth["operation"] != 0
            immediate = previous_type == "ask"
            result["operations_exact"] += int(exact)
            result["operations_total"] += 1
            for prefix, selected in (
                ("post_answer", seen_answer),
                ("meaningful_post_answer", seen_answer and meaningful),
                ("immediate_post_answer", immediate),
                ("meaningful_immediate_post_answer", immediate and meaningful),
            ):
                if selected:
                    result[prefix + "_operations_exact"] += int(exact)
                    result[prefix + "_operations_total"] += 1
            result["all_post_event_ledgers_exact"] &= ledger_exact
            row.update(expected_operation=expected, expected_post_event_ledger=truth["post_event_ledger"],
                       operation_exact=exact, ledger_exact=ledger_exact, after_answer=seen_answer,
                       immediately_after_answer=immediate, meaningful=meaningful)
        else:
            turn += 1
            query_exact = (observed["predicted_query_entity"] == truth["query_entity"] and
                           observed["predicted_query_plan"] == truth["query_plan"])
            keys_exact = observed["selected_keys"] == truth["supports"]
            values_exact = observed["selected_values"] == truth["evidence_values"]
            support_exact = keys_exact and values_exact
            answer_exact = observed["generated"] == truth["answer_text"]
            result["queries_exact"] += int(query_exact)
            result["queries_total"] += 1
            result["support_keys_exact"] += int(keys_exact)
            result["support_exact"] += int(support_exact)
            result["query_support_joint_exact"] += int(query_exact and support_exact)
            result["answers_exact"] += int(answer_exact)
            result["answers_total"] += 1
            result["generated_characters"] += observed["generated_characters"]
            result["control_characters"] += observed["control_characters"]
            result["answer_boundaries_reset"] += int(observed["state_reset_after_answer"])
            row.update(turn=turn, expected_query_entity=truth["query_entity"], expected_query_plan=truth["query_plan"],
                       expected_supports=truth["supports"], expected_evidence_values=truth["evidence_values"],
                       target=truth["answer_text"], query_exact=query_exact, support_keys_exact=keys_exact,
                       evidence_values_exact=values_exact, support_exact=support_exact,
                       query_support_joint_exact=query_exact and support_exact, answer_exact=answer_exact)
            seen_answer = True
        result["steps"].append(row)
        previous_type = truth["type"]
    result["all_operations_exact"] = result["operations_exact"] == result["operations_total"]
    result["all_queries_exact"] = result["queries_exact"] == result["queries_total"]
    result["all_supports_exact"] = result["support_exact"] == result["queries_total"]
    result["all_answers_exact"] = result["answers_exact"] == result["answers_total"]
    result["whole_world_exact"] = all(result[key] for key in ("all_operations_exact", "all_post_event_ledgers_exact",
                                                              "all_queries_exact", "all_supports_exact", "all_answers_exact"))
    result["final_ledger"] = trace[-1]["current_ledger"] if trace[-1]["type"] == "ask" else trace[-1]["post_event_ledger"]
    return result


def summarize(rows):
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    counters = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
    family_rows = []
    by_kind = {}
    for family_id, pair in families.items():
        if len(pair) != 2 or {row["side"] for row in pair} != {0, 1}:
            raise ValueError("a paired family must include both distinct worlds")
        pair.sort(key=lambda row: row["side"])
        questions = [[step for step in row["steps"] if step["type"] == "ask"] for row in pair]
        if len(questions[0]) != len(questions[1]):
            raise ValueError("paired question counts differ")
        turns = []
        for index, (left, right) in enumerate(zip(*questions)):
            turns.append({"turn": index + 1, "answer_changing": left["target"] != right["target"],
                          "both_answers_exact": left["answer_exact"] and right["answer_exact"],
                          "both_query_support_exact": left["query_support_joint_exact"] and right["query_support_joint_exact"]})
        fam = {"family_id": family_id, "kind": pair[0]["kind"],
               "changing_any_answer": any(turn["answer_changing"] for turn in turns),
               "both_worlds_exact": all(row["whole_world_exact"] for row in pair),
               "all_answers_exact": all(row["all_answers_exact"] for row in pair),
               "worlds_exact": sum(row["whole_world_exact"] for row in pair), "turns": turns,
               "answers_exact": sum(row["answers_exact"] for row in pair),
               "answers_total": sum(row["answers_total"] for row in pair),
               "meaningful_post_answer_operations_exact": sum(row["meaningful_post_answer_operations_exact"] for row in pair),
               "meaningful_post_answer_operations_total": sum(row["meaningful_post_answer_operations_total"] for row in pair)}
        family_rows.append(fam)
        stats = by_kind.setdefault(fam["kind"], {"families": 0, "both_worlds_exact": 0,
                                               "all_answers_exact": 0, "answers_exact": 0, "answers_total": 0})
        stats["families"] += 1
        for name in ("both_worlds_exact", "all_answers_exact", "answers_exact", "answers_total"):
            stats[name] += int(fam[name])
    turns = [turn for family in family_rows for turn in family["turns"]]
    paired_groups = {}
    for name, changing in (("answer_changing", True), ("invariant", False)):
        group = [turn for turn in turns if turn["answer_changing"] == changing]
        paired_groups[name] = {"question_pairs": len(group),
                               "both_answers_exact": sum(turn["both_answers_exact"] for turn in group),
                               "both_query_support_exact": sum(turn["both_query_support_exact"] for turn in group)}
    result = {"worlds": len(rows), "families": len(family_rows), **counters,
              "whole_worlds_exact": sum(row["whole_world_exact"] for row in rows),
              "whole_session_pairs_exact": sum(row["both_worlds_exact"] for row in family_rows),
              "complete_answer_session_pairs": sum(row["all_answers_exact"] for row in family_rows),
              "all_operations_exact_worlds": sum(row["all_operations_exact"] for row in rows),
              "all_ledgers_exact_worlds": sum(row["all_post_event_ledgers_exact"] for row in rows),
              "by_kind": by_kind, "paired_question_groups": paired_groups}
    denominator_metrics = {
        "answer_accuracy": ("answers_exact", "answers_total"),
        "query_support_joint_accuracy": ("query_support_joint_exact", "queries_total"),
        "meaningful_post_answer_operation_accuracy": ("meaningful_post_answer_operations_exact", "meaningful_post_answer_operations_total"),
        "immediate_post_answer_operation_accuracy": ("immediate_post_answer_operations_exact", "immediate_post_answer_operations_total"),
        "meaningful_immediate_post_answer_operation_accuracy": ("meaningful_immediate_post_answer_operations_exact", "meaningful_immediate_post_answer_operations_total"),
        "whole_world_accuracy": ("whole_worlds_exact", "worlds"),
        "whole_session_pair_accuracy": ("whole_session_pairs_exact", "families"),
    }
    result["rates"] = {name: result[num] / result[den] if result[den] else None
                       for name, (num, den) in denominator_metrics.items()}
    return result, family_rows


def evaluate_panel(model, families, *, batch_families=4, reset_after_answer=False,
                   memory_enabled=None, reader="strict", diagnostics=False):
    if diagnostics:
        raise ValueError("no oracle diagnostics are silently substituted into 002 candidate rollout")
    if batch_families < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    # A panel may contain different declared schedules. Batch equal schedules
    # without changing family identity or silently padding extra observations.
    groups = defaultdict(list)
    for original_index, family in enumerate(families):
        schedule = tuple(step["type"] for step in family["worlds"][0]["steps"])
        groups[schedule].append((original_index, family))
    scored = []
    for group in groups.values():
        for start in range(0, len(group), batch_families):
            indexed = group[start:start + batch_families]
            batch = [family for _, family in indexed]
            references = [world for family in batch for world in family["worlds"]]
            inputs = [candidate_inputs(world) for world in references]
            observed = rollout_batch(model, inputs, reset_after_answer=reset_after_answer,
                                     memory_enabled=memory_enabled, reader=reader)
            for index, (reference, trace) in enumerate(zip(references, observed["traces"])):
                source_index, family = indexed[index // 2]
                row = score_world(reference, trace, family_id=family["family_id"], side=index % 2,
                                  kind=family["kind"], wording=family.get("wording", "unspecified"))
                scored.append((source_index, index % 2, row))
    rows = [row for _, _, row in sorted(scored, key=lambda item: item[:2])]
    summary, family_rows = summarize(rows)
    return {"summary": summary, "family_rows": family_rows, "rows": rows,
            "policy": {"reader": reader, "reset_after_answer": bool(reset_after_answer),
                       "memory_enabled_override": memory_enabled,
                       "candidate_inputs": "explicit observe/ask boundary and text only",
                       "state_carry": "all emitted characters including period; no extra newline",
                       "support_exact": "selected record identities AND values match reference packet",
                       "labels": "used only after each complete candidate world rollout"}}


def run(args):
    if args.output.exists():
        raise FileExistsError("completed evaluation output must not be overwritten")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = json.loads(args.data.read_text())
    families = payload["families"] if isinstance(payload, dict) else payload
    model = FusionModel.from_checkpoint(args.checkpoint).eval()
    output = evaluate_panel(model, families, batch_families=args.batch_families,
                            reset_after_answer=args.reset_after_answer,
                            memory_enabled=False if args.memory_off else None, reader=args.reader)
    output.update(schema="cog-fusion-002-evaluation-v1", checkpoint_sha256=sha256(args.checkpoint),
                  data_sha256=sha256(args.data), family_sha256=digest(families),
                  model_identity=model.model_identity(), parameters=model.parameter_summary(),
                  neural_state_tensor_bytes=state_bytes(model.init_state(1)),
                  source_sha256={str(path.relative_to(HERE.parent)): sha256(path) for path in
                                 (Path(__file__), HERE / "data.py", HERE / "common.py",
                                  PARENT / "fusion/model.py", PARENT / "fusion/engine.py")})
    write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "policy": output["policy"], "summary": output["summary"]}), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-families", type=int, default=4)
    parser.add_argument("--reset-after-answer", action="store_true")
    parser.add_argument("--memory-off", action="store_true")
    parser.add_argument("--reader", choices=("strict", "hybrid"), default="strict")
    run(parser.parse_args())
