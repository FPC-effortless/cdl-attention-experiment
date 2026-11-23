"""Check the already declared live scenario without scoring any model.

Typed labels are used only for independent admission of the scorer. The live
runtime receives text, request IDs, and the explicitly declared policy controls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import PARENT, sha256, write_json
from data import semantic_signature
from fusion.data import ENTITIES, RELATIONS, UNKNOWN, TRAIN_TEMPLATES
from runtime_eval import new_scenario, declared_protocol


def audit():
    protocol_path = HERE / "evidence/RUNTIME_PROTOCOL.json"
    assert json.loads(protocol_path.read_text()) == declared_protocol()
    corpus_path = HERE / "evidence/data/corpus.json"
    corpus = json.loads(corpus_path.read_text())
    chars = set(json.loads((PARENT / "evidence/reproduction_inputs/tokenizer.json").read_text())["chars"])
    scenario = new_scenario()
    ledger, sources, pending = {}, {}, set()
    converted, first_three = [], []
    asks, observations, ask_schedule = 0, 0, []
    unknown_characters, unsupported_templates, answer_disagreements = [], [], []
    for step in scenario:
        if "text" in step:
            unknown_characters.extend(sorted(set(step["text"] + "\n") - chars))
        if step["kind"] == "observe":
            observations += 1
            op, entity, relation, value = step["expected_op"]
            expected = {"type": "observe", "operation": op,
                        "entity": ENTITIES.index(entity) if entity is not None else 0,
                        "relation": RELATIONS.index(relation) if relation is not None else 0,
                        "value": ENTITIES.index(value) if value is not None else UNKNOWN}
            converted.append(expected)
            if asks < 3:
                first_three.append(expected)
            if op:
                key = (entity, relation)
                if op == 1:
                    ledger[key], sources[key] = value, step["id"]
                else:
                    ledger.pop(key, None)
                    sources.pop(key, None)
            pending.difference_update(step.get("clear_pending", ()))
        elif step["kind"] == "ask":
            asks += 1
            ask_schedule.append(observations)
            entity, plan = ("vault", 0) if step["id"] == "unrelated-during-pending" else ("gem", 2)
            expected = {"type": "ask", "query_entity": ENTITIES.index(entity), "query_plan": plan}
            converted.append(expected)
            if asks <= 3:
                first_three.append(expected)
            if plan == 2:
                linked = ledger.get((entity, "container"))
                answer = ledger.get((linked, "owner"), "unknown")
            else:
                linked = entity
                answer = ledger.get((entity, "owner"), "unknown")
            expected_answer = "pending" if pending and linked == "box" else answer + "."
            if step["expected"] != expected_answer:
                answer_disagreements.append({"id": step["id"], "declared": step["expected"], "replayed": expected_answer})
        elif step["kind"] == "pending":
            assert step["key"] == ["world", "box", "owner"]
            pending.add(step["id"])
        elif step["kind"] == "erase_source":
            for key in [key for key, source in sources.items() if source == step["source"]]:
                ledger.pop(key, None)
                sources.pop(key, None)

    # All text is familiar wording. Enumerating possible identities is only a
    # template admission check and never produces operations for inference.
    familiar_strings = {
        template.format(e=entity, v=value)
        for templates in TRAIN_TEMPLATES.values() for template in templates
        for entity in ENTITIES for value in ENTITIES
    }
    for step in scenario:
        if "text" in step and step["text"] not in familiar_strings:
            unsupported_templates.append(step["id"])
    signatures = {"complete_learned_history": semantic_signature({"steps": converted}),
                  "first_three_queries": semantic_signature({"steps": first_three})}
    overlap = {}
    for split, families in corpus.items():
        observed = {semantic_signature(world) for family in families for world in family["worlds"]}
        overlap[split] = {name: signature in observed for name, signature in signatures.items()}
    result = {
        "schema": "cog-fusion-002-runtime-scenario-admission-v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "data/scorer admission only; no learned model inference or training",
        "source_sha256": sha256(Path(__file__)),
        "runtime_protocol_sha256": sha256(protocol_path),
        "corpus_sha256": sha256(corpus_path),
        "runtime_source_sha256": sha256(HERE / "runtime_eval.py"),
        "first_asks_after_observations": ask_schedule[:3],
        "all_asks_after_observations": ask_schedule,
        "observation_count": observations,
        "question_count_including_policy": asks,
        "unknown_characters": sorted(set(unknown_characters)),
        "unfamiliar_wording_step_ids": unsupported_templates,
        "declared_answer_disagreements": answer_disagreements,
        "semantic_signatures": signatures,
        "semantic_overlap_by_split": overlap,
        "notes": ["The complete sequence includes typed pending and source-erasure controls that are not language predictions.",
                  "Only the first three queries are used to establish the declared 2/4/6 schedule shift.",
                  "No accuracy threshold is added by this engineering admission check."]}
    result["pass"] = (ask_schedule[:3] == [2, 4, 6] and not unknown_characters
                      and not unsupported_templates and not answer_disagreements
                      and not any(any(values.values()) for values in overlap.values()))
    return result


if __name__ == "__main__":
    path = HERE / "review/RUNTIME_SCENARIO_ADMISSION.json"
    if path.exists():
        raise FileExistsError("preserve the first admission receipt")
    result = audit()
    write_json(path, result)
    print(json.dumps({"pass": result["pass"], "path": str(path)}))
    if not result["pass"]:
        raise SystemExit(1)
