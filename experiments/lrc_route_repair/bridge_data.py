"""Bridge-sensitive counterfactual diagnostic for the fixed VB-001 task.

Within a quartet, change ONLY the queried partner fact's person value.
All four people have different locations in the queried era, so the correct
answer must change with the first-hop bridge. No semantic metadata enters any
model forward. The parser below is a dataset validator only.

Copied without generator/validator changes from LRC-DIAG-001. New evaluation
seeds are declared in the caller's prospective protocol and may be generated
only after its fitting barrier has closed. The legacy qualify() entrypoint is
retained for provenance and is not called by this experiment.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from lrc_vb001_comparison.source import study as vb


def bridge_dataset(seed: int, groups: int):
    if not isinstance(seed, int) or not isinstance(groups, int) or groups < 1:
        raise ValueError("integer seed and positive integer group count required")
    rng = random.Random(seed)
    examples = []
    for _ in range(groups):
        entity, era = rng.randrange(4), rng.randrange(2)
        locations = list(range(4))
        rng.shuffle(locations)
        facts = []
        for e in range(2):
            block = []
            for person in range(4):
                block.append((e, person, 0, rng.randrange(4)))
                block.append((e, person, 1,
                              locations[person] if e == era else rng.randrange(4)))
            rng.shuffle(block)
            facts.extend(block)
        first = next(i for i, f in enumerate(facts) if f[:3] == (era, entity, 0))
        question = vb.encode(["During", "the", vb.ERAS[era], "era", "where", "did",
                              vb.NAMES[entity], "'s", "partner", "live", "?"])
        for bridge in range(4):
            variant = list(facts)
            variant[first] = (era, entity, 0, bridge)
            second = next(i for i, f in enumerate(variant) if f[:3] == (era, bridge, 1))
            records = []
            for e, person, relation, value in variant:
                words = ["During", "the", vb.ERAS[e], "era", vb.NAMES[person]]
                words += (["partnered", "with", vb.NAMES[value]] if relation == 0
                          else ["lived", "in", vb.PLACES[value]])
                records.append(vb.encode(words + ["."]))
            examples.append(dict(records=records, query=question, answer=locations[bridge],
                                 support=[first, second], era=era, entity=entity, person=bridge))
    return {key: torch.tensor([row[key] for row in examples], dtype=torch.long)
            for key in examples[0]}


dataset = bridge_dataset


def _parse_record(tokens):
    values = tokens.tolist()
    if len(values) != vb.MAX_WORDS or any(v < 0 or v >= len(vb.WORDS) for v in values):
        raise ValueError("invalid record vocabulary or length")
    try:
        era = tuple(vb.VOCAB[x] for x in vb.ERAS).index(values[2])
        person = tuple(vb.VOCAB[x] for x in vb.NAMES).index(values[4])
    except ValueError as exc:
        raise ValueError("invalid era or subject") from exc
    if values[5] == vb.VOCAB["partnered"]:
        relation, labels, suffix = 0, vb.NAMES, ["partnered", "with"]
    elif values[5] == vb.VOCAB["lived"]:
        relation, labels, suffix = 1, vb.PLACES, ["lived", "in"]
    else:
        raise ValueError("invalid relation")
    try:
        value = tuple(vb.VOCAB[x] for x in labels).index(values[7])
    except ValueError as exc:
        raise ValueError("invalid relation value") from exc
    expected = vb.encode(["During", "the", vb.ERAS[era], "era", vb.NAMES[person]]
                         + suffix + [labels[value], "."])
    if values != expected:
        raise ValueError("malformed record template or padding")
    return era, person, relation, value


def validate_bridge_dataset(data):
    keys = {"records", "query", "answer", "support", "era", "entity", "person"}
    if set(data) != keys or any(v.dtype != torch.long for v in data.values()):
        raise ValueError("wrong tensor keys or dtypes")
    n = len(data["answer"])
    expected_shapes = {"records": (n, vb.N_RECORDS, vb.MAX_WORDS),
                       "query": (n, vb.MAX_WORDS), "support": (n, 2),
                       **{k: (n,) for k in ("answer", "era", "entity", "person")}}
    if n == 0 or n % 4 or any(tuple(data[k].shape) != shape for k, shape in expected_shapes.items()):
        raise ValueError("invalid quartet or tensor shapes")
    changed_token_count = 0
    for start in range(0, n, 4):
        recs = data["records"][start:start + 4]
        query = data["query"][start:start + 4]
        support = data["support"][start:start + 4]
        era, entity = int(data["era"][start]), int(data["entity"][start])
        if not 0 <= era < 2 or not 0 <= entity < 4:
            raise ValueError("invalid metadata range")
        if not torch.all(data["era"][start:start + 4] == era) or not torch.all(data["entity"][start:start + 4] == entity):
            raise ValueError("query metadata changes within quartet")
        question = torch.tensor(vb.encode(["During", "the", vb.ERAS[era], "era", "where", "did",
                                          vb.NAMES[entity], "'s", "partner", "live", "?"]))
        if not torch.all(query == question[None]):
            raise ValueError("query changes or disagrees with metadata")
        if not torch.equal(data["person"][start:start + 4], torch.arange(4)):
            raise ValueError("quartet must visit every bridge person")
        if set(data["answer"][start:start + 4].tolist()) != set(range(4)):
            raise ValueError("quartet answers are not a four-class permutation")
        if not torch.all(support[:, 0] == support[0, 0]):
            raise ValueError("first support changes occurrence")
        if len(set(support[:, 1].tolist())) != 4:
            raise ValueError("second support must visit four distinct location occurrences")
        first = int(support[0, 0])
        if not 0 <= first < vb.N_RECORDS:
            raise ValueError("first support out of range")
        for bridge in range(4):
            differences = recs[bridge] != recs[0]
            allowed = torch.zeros_like(differences)
            if bridge:
                allowed[first, 7] = True
            if not torch.equal(differences, allowed):
                raise ValueError("a non-bridge token changes, or bridge value fails to change")
            changed_token_count += int(differences.sum())
            parsed = [_parse_record(r) for r in recs[bridge]]
            expected_keys = {(e, p, rel) for e in range(2) for p in range(4) for rel in range(2)}
            if {f[:3] for f in parsed} != expected_keys or len(set(f[:3] for f in parsed)) != 16:
                raise ValueError("missing or duplicate logical fact")
            if any(f[0] != i // 8 for i, f in enumerate(parsed)):
                raise ValueError("records do not preserve two era blocks")
            lookup = {f[:3]: (i, f[3]) for i, f in enumerate(parsed)}
            expected_first, person = lookup[era, entity, 0]
            expected_second, answer = lookup[era, person, 1]
            if person != bridge or support[bridge].tolist() != [expected_first, expected_second]:
                raise ValueError("support labels disagree with the two-hop rule")
            if int(data["answer"][start + bridge]) != answer:
                raise ValueError("answer label disagrees with selected location")
            if len({lookup[era, p, 1][1] for p in range(4)}) != 4:
                raise ValueError("queried-era locations are not all distinct")
    return {"status": "passed", "examples": n, "quartets": n // 4,
            "only_bridge_token_changes": True, "changed_tokens_vs_quartet_reference": changed_token_count,
            "query_and_other_facts_fixed": True, "all_four_answers_per_quartet": True,
            "unique_gold_supports_and_rule": True, "same_schema_as_vb_dataset": True,
            "data_sha256": vb.digest_data(data)}


validate = validate_bridge_dataset


def qualify():
    here = Path(__file__).resolve().parent
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_path = here / "bridge_qualification_plan.json"
    receipt_path = here / "bridge_qualification.json"
    if plan_path.exists() or receipt_path.exists():
        raise RuntimeError("Refusing to overwrite bridge qualification")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 99001,
            "groups": 8, "script_sha256": script_hash, "heldout_generated": False,
            "checks": ["full structural validator", "deterministic replay", "negative fixtures"]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    data = bridge_dataset(99001, 8)
    receipt = validate_bridge_dataset(data)
    assert vb.digest_data(data) == vb.digest_data(bridge_dataset(99001, 8))
    failures = []
    for name in ("wrong_answer", "wrong_second_support", "changed_query", "changed_other_fact"):
        bad = {k: v.clone() for k, v in data.items()}
        if name == "wrong_answer":
            bad["answer"][0] = (bad["answer"][0] + 1) % 4
        elif name == "wrong_second_support":
            bad["support"][0, 1] = (bad["support"][0, 1] + 1) % 16
        elif name == "changed_query":
            bad["query"][1, 6] = vb.VOCAB[vb.NAMES[(int(bad["entity"][1]) + 1) % 4]]
        else:
            j = (int(bad["support"][0, 0]) + 1) % 16
            labels = vb.NAMES if bad["records"][1, j, 5] == vb.VOCAB["partnered"] else vb.PLACES
            ids = [vb.VOCAB[x] for x in labels]
            bad["records"][1, j, 7] = ids[(ids.index(int(bad["records"][1, j, 7])) + 1) % 4]
        try:
            validate_bridge_dataset(bad)
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError(f"validator accepted corrupted fixture: {name}")
    receipt.update(seed=99001, groups=8, script_sha256=script_hash, deterministic_replay=True,
                   rejected_negative_fixtures=failures, heldout_generated=False,
                   closed_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    qualify()
