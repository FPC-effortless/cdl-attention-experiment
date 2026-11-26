"""Generated, paired lifecycle histories with explicit split and wording families.

Labels are supervision/evaluation only. Candidate inference receives text alone.
No claim of natural-language coverage follows from these controlled templates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

ENTITIES = ("Amara", "Bela", "Cora", "Davi", "Eren", "Faye", "Gita", "Hadi",
            "chest", "crate", "box", "vault", "coin", "gem", "ring", "key")
RELATIONS = ("owner", "container")
UNKNOWN = 16
KINDS = ("direct", "twohop", "correction", "irrelevant", "retraction", "bridge", "missing", "cancelled")

TRAIN_TEMPLATES = {
    "owner": ("{v} owns the {e}.", "The {e} belongs to {v}.", "The owner of the {e} is {v}."),
    "container": ("The {e} is in the {v}.", "The {v} contains the {e}.", "The {e} is kept inside the {v}."),
    "delete": ("The owner of the {e} is unknown.", "Forget who owns the {e}.", "Retract the owner of the {e}."),
    "cancelled": ("The {e} might belong to {v}, but no change happened.",
                  "A transfer of the {e} to {v} was cancelled.",
                  "They planned to give the {e} to {v}, then cancelled it."),
    "q_owner": ("Who owns the {e}?", "Who is the owner of the {e}?", "To whom does the {e} belong?"),
    "q_container": ("Where is the {e}?", "What contains the {e}?", "In which container is the {e}?"),
    "q_twohop": ("Who owns the container holding the {e}?", "Who owns what contains the {e}?",
                 "Who is the owner of the container with the {e}?"),
}
FRESH_TEMPLATES = {
    "owner": ("Ownership of the {e} rests with {v}.", "For the {e}, the owner is {v}."),
    "container": ("Inside the {v} lies the {e}.", "The {e} has the {v} as its container."),
    "delete": ("The ownership record for the {e} has been withdrawn.", "There is no accepted owner for the {e}."),
    "cancelled": ("Giving the {e} to {v} never took place.", "The proposal that {v} own the {e} was abandoned."),
    "q_owner": ("Name the person who possesses the {e}.", "Identify the current owner of the {e}."),
    "q_container": ("Name the container that holds the {e}.", "Identify the container of the {e}."),
    "q_twohop": ("Name the owner of whatever holds the {e}.", "Identify who owns the thing containing the {e}."),
}
FILLERS = ("The clock ticks.", "A bell rings.", "The day is quiet.")


def canonical(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def digest(x) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


@dataclass(frozen=True)
class Event:
    text: str
    operation: int  # 0 hold, 1 set, 2 delete
    entity: int = 0
    relation: int = 0
    value: int = UNKNOWN


def apply_event(ledger: dict[tuple[int, int], int], event: Event) -> None:
    if event.operation == 1:
        ledger[(event.entity, event.relation)] = event.value
    elif event.operation == 2:
        ledger.pop((event.entity, event.relation), None)


def answer(ledger: dict[tuple[int, int], int], entity: int, plan: int):
    relation = 1 if plan in (1, 2) else 0
    keys = [(entity, relation)]
    first = ledger.get(keys[0], UNKNOWN)
    values = [first]
    if plan == 2 and first != UNKNOWN:
        keys.append((first, 0))
        values.append(ledger.get(keys[-1], UNKNOWN))
    elif plan == 2:
        values.append(UNKNOWN)
    return values[-1], keys, values


def answer_text(value: int) -> str:
    return ("unknown" if value == UNKNOWN else ENTITIES[value]) + "."


def family(index: int, split: str, *, fresh: bool = False, extra_fillers: int = 0) -> dict:
    # No split reuses the same family-generation stream. Label assignment never
    # depends on split or wording. Only evaluation calls fresh=True.
    split_codes = {"train": 11, "dev": 23, "test": 47, "fresh": 59, "long": 71}
    rng = random.Random(71000019 * split_codes[split] + index * 103)
    templates = FRESH_TEMPLATES if fresh else TRAIN_TEMPLATES
    kind = KINDS[index % len(KINDS)]
    owners = rng.sample(range(8), 5)
    c, c2, unrelated, unused = rng.sample(range(8, 12), 4)
    item, missing_item = rng.sample(range(12, 16), 2)
    initial, new_a, new_b, other, offroute = owners
    style = {k: rng.randrange(len(v)) for k, v in templates.items()}
    filler = FILLERS[rng.randrange(len(FILLERS))]
    plan = 0 if kind == "direct" else 2
    if kind in ("irrelevant", "missing") and (index // len(KINDS)) % 2:
        plan = 1
    query_entity = c if plan == 0 else item
    if kind == "missing":
        query_entity = missing_item

    def make(t, e, v=UNKNOWN, op=1, relation=0):
        text = templates[t][style[t]].format(e=ENTITIES[e], v=ENTITIES[v] if v != UNKNOWN else "unknown")
        return Event(text, op, e, relation, v)

    worlds = []
    for side in (0, 1):
        target_owner = (new_a, new_b)[side] if kind in ("direct", "twohop") else initial
        events = [make("owner", c, target_owner), make("container", item, c, relation=1),
                  make("owner", c2, other)]
        if kind == "correction":
            change = make("owner", c, (new_a, new_b)[side])
        elif kind == "irrelevant":
            change = make("owner", unrelated, (new_a, new_b)[side])
        elif kind == "retraction":
            change = make("delete", c, op=2) if side == 0 else make("owner", c, new_b)
        elif kind == "bridge":
            change = make("container", item, c if side == 0 else c2, relation=1)
        elif kind == "cancelled":
            change = make("cancelled", c, (new_a, new_b)[side], op=0)
        else:
            change = Event(filler, 0)
        events.append(change)
        events.extend(Event(FILLERS[j % len(FILLERS)], 0) for j in range(extra_fillers))
        # A final off-route value makes last-visible-value copying fail.
        events.append(make("owner", unused, (new_a, new_b)[side] if kind == "missing" else offroute))
        q_kind = ("q_owner", "q_container", "q_twohop")[plan]
        query = templates[q_kind][style[q_kind]].format(e=ENTITIES[query_entity])
        ledger = {}
        ledgers = []
        for ev in events:
            apply_event(ledger, ev)
            ledgers.append([[e, r, v] for (e, r), v in sorted(ledger.items())])
        result, supports, values = answer(ledger, query_entity, plan)
        worlds.append({"events": [asdict(e) for e in events], "query": query,
                       "query_entity": query_entity, "query_plan": plan,
                       "answer": result, "answer_text": answer_text(result),
                       "supports": supports, "evidence_values": values,
                       "post_event_ledgers": ledgers})
    return {"family_id": f"{split}-{index:05}", "kind": kind,
            "wording": "held_lexical_and_syntactic_wording" if fresh else "familiar_templates", "worlds": worlds}


def build_corpus(path: Path, train_families: int = 1024, held_families: int = 128) -> dict:
    data = {}
    previous_semantics = set()
    regenerated = 0
    def semantic_signature(world):
        operations = [(ev["operation"], ev["entity"], ev["relation"],
                       ev["value"] if ev["operation"] == 1 else UNKNOWN)
                      for ev in world["events"] if ev["operation"]]
        return digest([operations, world["query_entity"], world["query_plan"]])
    for split, count in (("train", train_families), ("dev", 64), ("test", held_families),
                         ("fresh", held_families), ("long", 32)):
        items = []
        this_semantics = set()
        for i in range(count):
            attempt = 0
            while True:
                fam = family(i + attempt * 100000, split, fresh=split == "fresh",
                             extra_fillers=64 if split == "long" else 0)
                sigs = {semantic_signature(w) for w in fam["worlds"]}
                if not sigs.intersection(previous_semantics):
                    break
                attempt += 1
                regenerated += 1
                if attempt > 100:
                    raise RuntimeError("semantic dedup regeneration exhausted")
            fam["family_id"] = f"{split}-{i:05}"
            items.append(fam)
            this_semantics.update(sigs)
        data[split] = items
        previous_semantics.update(this_semantics)
    # Whole-history duplicate leakage is rejected. Individual training phrases
    # and entity names are intentionally shared; syntax OOD is a separate panel.
    seen = {}
    for split, families in data.items():
        for fam in families:
            for world in fam["worlds"]:
                key = digest({"events": [e["text"] for e in world["events"]], "query": world["query"]})
                if key in seen and seen[key] != split:
                    raise ValueError(f"cross-split duplicate history: {seen[key]} / {split}")
                seen[key] = split
            if digest(fam["worlds"][0]["events"]) == digest(fam["worlds"][1]["events"]):
                raise ValueError("counterfactual family contains identical event worlds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(data))
    manifest = {"schema": "cog-fusion-data-v1", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "entities": ENTITIES, "relations": RELATIONS, "unknown": UNKNOWN,
                "family_counts": {k: len(v) for k, v in data.items()},
                "world_counts": {k: 2 * len(v) for k, v in data.items()},
                "split_hashes": {k: digest(v) for k, v in data.items()},
                "training_templates": TRAIN_TEMPLATES, "held_syntax_templates": FRESH_TEMPLATES,
                "cross_split_duplicate_histories": 0,
                "cross_split_duplicate_semantic_worlds": 0, "regenerated_families": regenerated,
                "pair_counts": {k: {"answer_changing": sum(f["worlds"][0]["answer"] != f["worlds"][1]["answer"] for f in v),
                                    "invariant_or_unknown": sum(f["worlds"][0]["answer"] == f["worlds"][1]["answer"] for f in v)}
                                for k, v in data.items()},
                "limits": ["fixed 16-identity vocabulary", "controlled synthetic language",
                           "held wording combines lexical and syntactic shift; entity names stay fixed",
                           "long history has extra neutral filler, not new reasoning structure"]}
    path.with_name("DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.path), indent=2))
