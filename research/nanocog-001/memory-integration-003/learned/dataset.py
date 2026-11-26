"""Prospective synthetic histories; generation never calls the learned model.

Semantic family lineages split before rendering. Fresh syntax families are
absent from TRAIN and DEV. The generator exposes labels ONLY to the evaluator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from nanocog.memory.learned_archive import BOXES, ITEMS, OWNERS, UNKNOWN, operation

DATA_SEED = 823197
LENGTHS = (8, 24, 64)
SPLITS = (("train", 512), ("dev", 128), ("familiar_test", 256), ("fresh_test", 256))
SYNTAX = {
    "canonical": {
        "owner_init": "{owner} holds the {box}.",
        "owner_write": "{source} gave the {box} to {owner}.",
        "owner_noop": "{source} did not give the {box} to {owner}.",
        "item_init": "The {item} is in the {box}.",
        "item_write": "{actor} put the {item} in the {box}.",
        "item_noop": "{actor} did not put the {item} in the {box}.",
        "query_box": "Who holds the {box}?",
        "query_item": "Who holds the box with the {item}?",
    },
    "matched_recipient": {
        "owner_init": "The {box} belongs to {owner}.",
        "owner_write": "{owner} received the {box} from {source}.",
        "owner_noop": "{owner} never received the {box} from {source}.",
        "item_init": "Inside the {box} is the {item}.",
        "item_write": "The {item} went into the {box}, placed by {actor}.",
        "item_noop": "The {item} never went into the {box}, despite {actor}'s plan.",
        "query_box": "Name the owner of the {box}.",
        "query_item": "Name the owner of the container of the {item}.",
    },
    "matched_completion": {
        "owner_init": "For the {box}, the owner is {owner}.",
        "owner_write": "The handover of the {box} from {source} to {owner} was completed.",
        "owner_noop": "The handover of the {box} from {source} to {owner} was cancelled.",
        "item_init": "The location of the {item} is the {box}.",
        "item_write": "{actor}'s placement of the {item} in the {box} was completed.",
        "item_noop": "{actor}'s placement of the {item} in the {box} was cancelled.",
        "query_box": "Which person owns the {box}?",
        "query_item": "Which person owns the container that contains the {item}?",
    },
    "fresh_passive_delivery": {
        "owner_init": "The {box} is held by {owner}.",
        "owner_write": "From {source}, the {box} was delivered to {owner}.",
        "owner_noop": "From {source}, the {box} was not delivered to {owner}.",
        "item_init": "The {item} is contained by the {box}.",
        "item_write": "Into the {box}, the {item} was deposited by {actor}.",
        "item_noop": "Into the {box}, the {item} was not deposited by {actor}.",
        "query_box": "By whom is the {box} held?",
        "query_item": "By whom is the container holding the {item} owned?",
    },
    "fresh_status_front": {
        "owner_init": "Current holder of the {box}: {owner}.",
        "owner_write": "Successful transfer: {source} to {owner}, carrying the {box}.",
        "owner_noop": "Aborted transfer: {source} to {owner}, carrying the {box}.",
        "item_init": "Current container of the {item}: the {box}.",
        "item_write": "Successful placement: {actor} moved the {item} into the {box}.",
        "item_noop": "Aborted placement: {actor} moved the {item} into the {box}.",
        "query_box": "Identify the current holder of the {box}.",
        "query_item": "Identify the current holder of the container with the {item}.",
    },
    "fresh_retention_contrast": {
        "owner_init": "Possession of the {box} rests with {owner}.",
        "owner_write": "{source} released the {box}; {owner} took possession.",
        "owner_noop": "{source} retained the {box}; {owner} took no possession.",
        "item_init": "The {item} rests within the {box}.",
        "item_write": "{actor} finished moving the {item}; its destination was the {box}.",
        "item_noop": "{actor} abandoned moving the {item}; its intended destination was the {box}.",
        "query_box": "Who has possession of the {box}?",
        "query_item": "Who has possession of the box containing the {item}?",
    },
    "fresh_destination_first": {
        "owner_init": "{owner} is responsible for holding the {box}.",
        "owner_write": "To {owner} from {source}: the {box} changed hands.",
        "owner_noop": "To {owner} from {source}: the {box} did not change hands.",
        "item_init": "The {box} is where the {item} can be found.",
        "item_write": "Into the {box} went the {item} when {actor} moved it.",
        "item_noop": "Into the {box} the {item} did not go when {actor} tried to move it.",
        "query_box": "Give the name of the person with the {box}.",
        "query_item": "Give the name of the person with the box around the {item}.",
    },
}
TRAIN_SYNTAX = tuple(SYNTAX)[:3]
FRESH_SYNTAX = tuple(SYNTAX)[3:]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def semantic_pair(split: str, index: int) -> dict:
    lineage = hashlib.sha256(f"{DATA_SEED}:{split}:{index}".encode()).hexdigest()
    rng = random.Random(int(lineage[:16], 16))
    length, route = LENGTHS[index % 3], index % 8
    target_box = route if route < 4 else rng.randrange(4)
    linked_item = route - 4 if route >= 4 else rng.randrange(4)
    state = [UNKNOWN] * 8
    events = []
    order = list(range(4)); rng.shuffle(order)
    for record in order:
        value = rng.randrange(8)
        events.append(dict(record=record, value=value, commit=True, initial=True,
                           source=0, actor=rng.randrange(8)))
        state[record] = value
    events.append(dict(record=4 + linked_item, value=8 + target_box, commit=True,
                       initial=True, source=0, actor=rng.randrange(8)))
    state[4 + linked_item] = 8 + target_box
    target_at = rng.randrange(5, length)
    for t in range(5, length):
        if t == target_at:
            record = target_box
        else:
            allowed = [r for r in range(8) if r != 4 + linked_item and
                       (t < target_at or r != target_box)]
            record = rng.choice(allowed)
        source = state[record] if state[record] != UNKNOWN else 0
        values = list(range(8)) if record < 4 else list(range(8, 12))
        values = [v for v in values if v != state[record]]
        value = rng.choice(values)
        commit = t == target_at or rng.random() >= .25
        events.append(dict(record=record, value=value, commit=commit, initial=False,
                           source=source, actor=rng.randrange(8)))
        if commit:
            state[record] = value
    return dict(lineage=lineage, split=split, index=index, length=length, route=route,
                target_at=target_at, target_record=target_box, events=events)


def render_pair(semantic: dict, syntax: str, view: int = 0) -> dict:
    templates = SYNTAX[syntax]
    # Global within-type permutations provide counterfactual role/value swaps.
    rng = random.Random(int(semantic["lineage"][:16], 16) + 7907 * view)
    owner_map, box_map, item_map = list(range(8)), list(range(4)), list(range(4))
    if view:
        rng.shuffle(owner_map); rng.shuffle(box_map); rng.shuffle(item_map)
    def remap_record(r):
        return box_map[r] if r < 4 else 4 + item_map[r - 4]
    def remap_value(v):
        return owner_map[v] if v < 8 else 8 + box_map[v - 8]
    route = remap_record(semantic["route"])
    worlds = []
    for world in range(2):
        ledger, events = [UNKNOWN] * 8, []
        for i, event in enumerate(semantic["events"]):
            r, v = remap_record(event["record"]), remap_value(event["value"])
            commit = event["commit"] and not (world == 1 and i == semantic["target_at"])
            owner_record = r < 4
            kind = ("owner" if owner_record else "item") + ("_init" if event["initial"] else
                      ("_write" if commit else "_noop"))
            fields = dict(owner=OWNERS[v] if owner_record else OWNERS[0],
                          source=OWNERS[owner_map[event["source"]]] if owner_record else OWNERS[0],
                          actor=OWNERS[owner_map[event["actor"]]],
                          box=BOXES[r] if owner_record else BOXES[v - 8],
                          item=ITEMS[r - 4] if not owner_record else ITEMS[0])
            text = templates[kind].format(**fields)
            op = operation(r, v) if commit else 0
            if commit:
                ledger[r] = v
            events.append(dict(text=text, operation=op, ledger=list(ledger)))
        query = templates["query_box" if route < 4 else "query_item"].format(
            box=BOXES[route] if route < 4 else BOXES[0],
            item=ITEMS[route - 4] if route >= 4 else ITEMS[0])
        final_value = ledger[route] if route < 4 else ledger[ledger[route] - 8]
        assert 0 <= final_value < 8
        worlds.append(dict(events=events, query=query, route=route, answer=final_value))
    assert worlds[0]["answer"] != worlds[1]["answer"]
    return dict(lineage=semantic["lineage"], split=semantic["split"], index=semantic["index"],
                length=semantic["length"], syntax=syntax, view=view,
                target_at=semantic["target_at"], target_record=remap_record(semantic["target_record"]),
                worlds=worlds)


def build() -> dict:
    data = {"control_train": [], "candidate_train": [], "dev": [],
            "familiar_test": [], "fresh_test": []}
    lineages = {}
    for split, count in SPLITS:
        lineages[split] = []
        for i in range(count):
            semantic = semantic_pair(split, i)
            lineages[split].append(semantic["lineage"])
            if split == "train":
                data["control_train"].append(render_pair(semantic, "canonical"))
                for view, syntax in enumerate(TRAIN_SYNTAX):
                    data["candidate_train"].append(render_pair(semantic, syntax, view))
            else:
                syntax = FRESH_SYNTAX[(i // 8) % 4] if split == "fresh_test" else "canonical"
                data[split].append(render_pair(semantic, syntax))
    assert len(set(sum(lineages.values(), []))) == sum(len(v) for v in lineages.values())
    return {"data": data, "lineages": lineages}


def pin(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    payload = build()
    (destination / "corpus.json").write_bytes(canonical(payload))
    manifest = {"schema": 1, "data_seed": DATA_SEED, "history_lengths": LENGTHS,
                "family_counts": dict(SPLITS), "syntax": SYNTAX,
                "train_syntax": TRAIN_SYNTAX, "fresh_syntax": FRESH_SYNTAX,
                "lineage_hashes": {k: sha(v) for k, v in payload["lineages"].items()},
                "split_hashes": {k: sha(v) for k, v in payload["data"].items()},
                "corpus_sha256": sha(payload),
                "unique_texts": {k: len({e["text"] for p in v for w in p["worlds"] for e in w["events"]} |
                                        {w["query"] for p in v for w in p["worlds"]}) for k, v in payload["data"].items()}}
    (destination / "DATA_MANIFEST.json").write_bytes(canonical(manifest))
    return manifest
