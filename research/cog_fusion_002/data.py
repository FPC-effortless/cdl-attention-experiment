"""Paired, interleaved observation/answer histories for COG-FUSION-002.

Generated labels describe the intended typed world. They are supervised training
targets or post-inference evaluation labels, never candidate inference inputs.
Every ordinary world has eight observations and asks after observations 3/5/8.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from common import PARENT, HERE
from fusion.data import (ENTITIES, RELATIONS, UNKNOWN, KINDS, FILLERS,
                         TRAIN_TEMPLATES as OLD_FAMILIAR, FRESH_TEMPLATES as OLD_FRESH,
                         answer as resolve_truth, answer_text)


SCHEMA = "cog-fusion-multiturn-data-v2"
NEW_TRAIN_TEMPLATES = {
    "owner": ("{v} is the owner of the {e}.", "The {e} is owned by {v}."),
    "container": ("The {e} sits inside the {v}.", "The {v} holds the {e}."),
    "delete": ("Remove the owner recorded for the {e}.", "The owner entry for the {e} is erased."),
    "cancelled": ("The {e} was not given to {v}; its owner stays the same.",
                  "The change making {v} owner of the {e} was cancelled."),
    "q_owner": ("Tell me who owns the {e}.", "Which person owns the {e}?"),
    "q_container": ("Which container holds the {e}?", "Tell me where the {e} is kept."),
    "q_twohop": ("Tell me who owns the container of the {e}.", "Who owns the place where the {e} is kept?"),
}
TRAIN_TEMPLATES = {kind: tuple(OLD_FAMILIAR[kind]) + tuple(NEW_TRAIN_TEMPLATES[kind]) for kind in OLD_FAMILIAR}
FRESH_TEMPLATES = {
    "owner": ("The {e} has passed into the ownership of {v}.", "{v} now has ownership of the {e}."),
    "container": ("The {e} can be found within the {v}.", "The {v} is where the {e} has been placed."),
    "delete": ("Disregard the former ownership information for the {e}.", "No owner is currently registered for the {e}."),
    "cancelled": ("The proposed handover of the {e} to {v} failed to occur.",
                  "{v} was going to receive the {e}, but the transfer was called off."),
    "q_owner": ("Give the name of the {e}'s owner.", "Whose property is the {e}?"),
    "q_container": ("What is the {e} located inside?", "Which object is holding the {e}?"),
    "q_twohop": ("Whose container has the {e} inside it?", "Who possesses the container in which the {e} lies?"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ledger_rows(ledger):
    return [[entity, relation, value] for (entity, relation), value in sorted(ledger.items())]


def execute_truth(ledger, step):
    if step["operation"] == 1:
        ledger[(step["entity"], step["relation"])] = step["value"]
    elif step["operation"] == 2:
        ledger.pop((step["entity"], step["relation"]), None)


def semantic_signature(world):
    # Ignore neutral/cancelled HOLD text when comparing accepted semantic worlds.
    # Within a cancellation pair, accepted states intentionally agree. Across
    # splits even that same accepted history is forbidden, irrespective of prose.
    sequence = []
    for step in world["steps"]:
        if step["type"] == "ask":
            sequence.append(("ask", step["query_entity"], step["query_plan"]))
        elif step["operation"]:
            sequence.append(("observe", step["operation"], step["entity"], step["relation"],
                             step["value"] if step["operation"] == 1 else UNKNOWN))
    return digest(sequence)


def text_signature(world):
    return digest([(step["type"], step["text"]) for step in world["steps"]])


def family(index: int, split: str, *, fresh=False, extra_fillers=0):
    split_codes = {"train": 103, "dev": 127, "test": 151, "fresh": 181, "long": 211}
    rng = random.Random(20071000019 * split_codes[split] + index * 1009)
    bank = FRESH_TEMPLATES if fresh else TRAIN_TEMPLATES
    kind = KINDS[index % len(KINDS)]
    initial, a, b, later_a, later_b, other, noise = rng.sample(range(8), 7)
    container, secondary, unrelated, unused = rng.sample(range(8, 12), 4)
    item, spare, missing = rng.sample(range(12, 16), 3)
    styles = {(position, typ): rng.randrange(len(bank[typ])) for position in range(11) for typ in bank}
    first_question = rng.choice(((container, 0), (item, 1), (item, 2), (secondary, 0)))
    if kind == "direct":
        first_question = (container, 0)
    elif kind == "twohop":
        first_question = (item, 2)
    elif kind == "missing":
        first_question = (missing, rng.randrange(3))
    if kind == "direct":
        middle_question = (container, 0)
    elif kind == "twohop":
        middle_question = (item, 2)
    elif kind == "bridge":
        middle_question = (item, rng.choice((1, 2)))
    elif kind == "missing":
        middle_question = (missing, rng.randrange(3))
    elif kind == "irrelevant":
        middle_question = rng.choice(((container, 0), (item, 1), (item, 2), (secondary, 0)))
    else:
        middle_question = rng.choice(((container, 0), (item, 2)))
    if kind == "direct":
        final_plan = 0
    elif kind == "twohop":
        final_plan = 2
    elif kind in ("bridge", "irrelevant", "missing"):
        final_plan = rng.randrange(3) if kind != "bridge" else rng.choice((1, 2))
    else:
        final_plan = rng.choice((0, 2))
    final_question = (missing if kind == "missing" else container if final_plan == 0 else item, final_plan)
    worlds = []
    for side in (0, 1):
        ledger, steps = {}, []

        def observe(typ, entity=0, value=UNKNOWN, *, operation=1, relation=0, position=0):
            if typ == "neutral":
                text = FILLERS[(index + position) % len(FILLERS)]
                operation, entity, relation, value = 0, 0, 0, UNKNOWN
                template_index = (index + position) % len(FILLERS)
            else:
                template_index = styles[position, typ]
                text = bank[typ][template_index].format(e=ENTITIES[entity],
                    v="unknown" if value == UNKNOWN else ENTITIES[value])
            step = {"type": "observe", "text": text, "operation": operation, "entity": entity,
                    "relation": relation, "value": value, "template_family": typ, "template_index": template_index}
            execute_truth(ledger, step)
            step["post_event_ledger"] = ledger_rows(ledger)
            steps.append(step)

        def ask(question, position):
            entity, plan = question
            typ = ("q_owner", "q_container", "q_twohop")[plan]
            template_index = styles[position, typ]
            text = bank[typ][template_index].format(e=ENTITIES[entity])
            value, supports, values = resolve_truth(ledger, entity, plan)
            steps.append({"type": "ask", "text": text, "query": text,
                          "query_entity": entity, "query_plan": plan, "answer": value,
                          "answer_text": answer_text(value), "supports": [list(key) for key in supports],
                          "evidence_values": values, "current_ledger": ledger_rows(ledger),
                          "template_family": typ, "template_index": template_index})

        first_owner = (a, b)[side] if kind in ("direct", "twohop") else initial
        observe("owner", container, first_owner, position=0)
        observe("container", item, container, relation=1, position=1)
        observe("owner", secondary, other, position=2)
        ask(first_question, 3)
        for filler in range(extra_fillers // 2):
            observe("neutral", position=filler)
        if kind in ("direct", "twohop"):
            observe("owner", container, (later_a, later_b)[side], position=4)
        elif kind == "correction":
            observe("owner", container, (a, b)[side], position=4)
        elif kind in ("irrelevant", "missing"):
            observe("owner", unrelated, (a, b)[side], position=4)
        elif kind == "retraction":
            observe("delete", container, operation=2, position=4) if side == 0 else observe("owner", container, a, position=4)
        elif kind == "bridge":
            observe("container", item, container if side == 0 else secondary, relation=1, position=4)
        elif kind == "cancelled":
            observe("cancelled", container, (a, b)[side], operation=0, position=4)
        # Familiar neutral HOLD is present in TRAIN, so long fillers do not add
        # unseen wording. The optional second observation remains off the route.
        if index % 3 == 0:
            observe("neutral", position=5)
        else:
            observe("owner", unused, noise, position=5)
        ask(middle_question, 6)
        for filler in range(extra_fillers - extra_fillers // 2):
            observe("neutral", position=100 + filler)
        if kind in ("direct", "twohop"):
            observe("owner", container, first_owner, position=7)
        elif kind == "correction":
            observe("owner", container, (later_a, later_b)[side], position=7)
        elif kind in ("irrelevant", "missing"):
            observe("owner", unrelated, (later_a, later_b)[side], position=7)
        elif kind == "retraction":
            observe("owner", container, b, position=7) if side == 0 else observe("delete", container, operation=2, position=7)
        elif kind == "bridge":
            observe("container", item, secondary if side == 0 else container, relation=1, position=7)
        elif kind == "cancelled":
            observe("cancelled", container, (later_a, later_b)[side], operation=0, position=7)
        if kind == "bridge":
            observe("owner", secondary, a, position=8)
        elif kind in ("twohop", "irrelevant", "missing", "cancelled"):
            observe("container", spare, secondary, relation=1, position=8)
        else:
            observe("cancelled", container, other, operation=0, position=8)
        observe("owner", unused, noise, position=9)
        ask(final_question, 10)
        worlds.append({"steps": steps, "final_ledger": ledger_rows(ledger),
                       "answers": [step["answer_text"] for step in steps if step["type"] == "ask"]})
    return {"family_id": f"{split}-{index:05}", "kind": kind,
            "wording": "new_held_lexical_and_syntactic_templates" if fresh else "familiar_training_templates",
            "worlds": worlds}


def validate_family(fam):
    if len(fam["worlds"]) != 2 or text_signature(fam["worlds"][0]) == text_signature(fam["worlds"][1]):
        raise ValueError("paired worlds must contain genuinely different input histories")
    schedules = []
    for world in fam["worlds"]:
        ledger, asks, observations = {}, 0, 0
        schedule = []
        for step in world["steps"]:
            schedule.append(step["type"])
            if step["type"] == "observe":
                execute_truth(ledger, step)
                observations += 1
                if ledger_rows(ledger) != step["post_event_ledger"]:
                    raise ValueError("incorrect post-observation ledger label")
            elif step["type"] == "ask":
                asks += 1
                result, keys, values = resolve_truth(ledger, step["query_entity"], step["query_plan"])
                expected = (result, answer_text(result), [list(key) for key in keys], values, ledger_rows(ledger))
                actual = tuple(step[key] for key in ("answer", "answer_text", "supports", "evidence_values", "current_ledger"))
                if actual != expected or step["query"] != step["text"]:
                    raise ValueError("incorrect query, support or answer label")
            else:
                raise ValueError("unknown transcript step")
        if asks != 3 or observations not in (8, 72) or ledger_rows(ledger) != world["final_ledger"]:
            raise ValueError("world violates transcript geometry or final ledger")
        schedules.append(schedule)
    if schedules[0] != schedules[1]:
        raise ValueError("paired world schedules differ")


def build_corpus(path: Path, train_families=2048, held_families=128, *, dev_families=64, long_families=32):
    train_patterns = set(sum((list(values) for values in TRAIN_TEMPLATES.values()), []))
    old_held = set(sum((list(values) for values in OLD_FRESH.values()), []))
    new_held = set(sum((list(values) for values in FRESH_TEMPLATES.values()), []))
    if train_patterns & old_held or train_patterns & new_held or old_held & new_held:
        raise ValueError("wording-bank contamination")
    data, all_semantics, all_texts = {}, set(), set()
    regenerated = 0
    for split, count in (("train", train_families), ("dev", dev_families), ("test", held_families),
                         ("fresh", held_families), ("long", long_families)):
        families = []
        for index in range(count):
            for attempt in range(101):
                fam = family(index + attempt * 100000, split, fresh=split == "fresh",
                             extra_fillers=64 if split == "long" else 0)
                signatures = {semantic_signature(world) for world in fam["worlds"]}
                texts = {text_signature(world) for world in fam["worlds"]}
                if not signatures & all_semantics and not texts & all_texts:
                    break
                regenerated += 1
            else:
                raise RuntimeError("deduplication regeneration budget exhausted")
            fam["family_id"] = f"{split}-{index:05}"
            validate_family(fam)
            families.append(fam)
            all_semantics.update(signatures)
            all_texts.update(texts)
        data[split] = families
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(data))
    split_files = {}
    for split, families in data.items():
        destination = path.with_name(split + ".json")
        destination.write_bytes(canonical({"schema": SCHEMA, "split": split, "families": families}))
        split_files[split] = {"filename": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    probe = data["train"][:min(128, len(data["train"]))]
    probe_path = path.with_name("train_probe.json")
    probe_path.write_bytes(canonical({"schema": SCHEMA, "split": "train_probe", "source_split": "train",
                                     "source_indices": list(range(len(probe))), "families": probe}))
    split_files["train_probe"] = {"filename": probe_path.name, "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest()}
    paired = {}
    distributions = {}
    for split, families in data.items():
        changing = sum(a != b for fam in families for a, b in zip(fam["worlds"][0]["answers"], fam["worlds"][1]["answers"]))
        session_changes = sum(fam["worlds"][0]["answers"] != fam["worlds"][1]["answers"] for fam in families)
        paired[split] = {"question_pairs_answer_changing": changing,
                         "question_pairs_invariant": len(families) * 3 - changing,
                         "session_pairs_with_any_answer_change": session_changes,
                         "session_pairs_with_all_answers_invariant": len(families) - session_changes}
        turns = []
        for turn in range(3):
            questions = [[step for step in world["steps"] if step["type"] == "ask"][turn]
                         for fam in families for world in fam["worlds"]]
            turns.append({"turn": turn + 1,
                          "query_plan_counts": dict(Counter(str(question["query_plan"]) for question in questions)),
                          "query_entity_counts": dict(Counter(ENTITIES[question["query_entity"]] for question in questions)),
                          "answer_character_lengths": dict(Counter(str(len(question["answer_text"])) for question in questions))})
        distributions[split] = turns
    manifest = {"schema": SCHEMA, "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "family_counts": {split: len(families) for split, families in data.items()},
                "world_counts": {split: 2 * len(families) for split, families in data.items()},
                "split_family_sha256": {split: digest(families) for split, families in data.items()},
                "split_files": split_files, "entities": ENTITIES, "relations": RELATIONS, "unknown": UNKNOWN,
                "schedule": "eight observations; asks after observations 3,5,8; no extra newline after generated answers",
                "long_schedule": "32 neutral observations after each of the first two asks, in addition to ordinary schedule",
                "templates_train": TRAIN_TEMPLATES, "templates_new_train": NEW_TRAIN_TEMPLATES,
                "templates_new_held": FRESH_TEMPLATES, "old_held_templates_excluded_from_train": OLD_FRESH,
                "cross_split_duplicate_text_histories": 0, "cross_split_duplicate_semantic_histories": 0,
                "within_split_duplicate_semantic_histories_across_families": 0,
                "identical_input_pairs": 0, "regenerated_families": regenerated,
                "paired_outcomes": paired, "question_distributions_by_turn": distributions,
                "train_probe_family_indices": list(range(len(probe))),
                "limits": ["fixed sixteen-identity vocabulary and controlled language templates",
                           "held wording changes both lexicon and syntax; no new-entity vocabulary test",
                           "cancellation pairs intentionally share accepted states while their proposed edits differ",
                           "old exposed scenarios are not inserted as training examples; familiar templates and names are shared",
                           "long panel extends neutral sequence length; it is not broad long-horizon reasoning"]}
    path.with_name("DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=HERE / "evidence/data/corpus.json")
    args = parser.parse_args()
    result = build_corpus(args.path)
    print(json.dumps({"path": str(args.path), "family_counts": result["family_counts"],
                      "corpus_sha256": result["corpus_sha256"], "regenerated_families": result["regenerated_families"]}, indent=2))
