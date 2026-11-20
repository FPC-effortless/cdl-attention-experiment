"""Compare citation-only and resolver-aware answer reuse on public CompanyWorld fixtures."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from investigation_world.benchmark.policies import PublicEvidenceReferencePolicy
from investigation_world.companyworld.verifier import verify_companyworld
from research_lab.ecosystem import _fixture

TYPES = {"INVESTIGATE_MISSING_SHIPMENT": "carrier_manifest",
         "INVESTIGATE_AUTHORITY_BREACH": "policy_rule"}
CHANGES = ("unchanged", "projection", "revision", "supersession", "restored_precedence",
           "unrelated_object", "lower_priority_insert", "unobservable_override")


def fingerprint(payload, initial, arm):
    if arm == "control":
        cited = {entry["record_id"] for entry in initial.evidence}
        dependencies = [r for r in payload["records"] if r["record_id"] in cited]
    else:
        task = payload["task"]
        kind = TYPES[task["task_type"]]
        target = task["target_object_id"]
        dependencies = next((r for r in payload["records"]
                             if r["record_type"] == kind and r["object_id"] == target), None)
    return hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).digest()


def fixture_pair(index, increment, change):
    original = _fixture(index)
    src = original.records[0]
    other = src.model_copy(deep=True)
    other.record_id += "-ALTERNATE"
    fact = original.oracle.facts[0]
    other.fields[fact.field_name] = fact.expected_value + increment
    if change in ("restored_precedence", "unobservable_override"):
        original.records.append(other)
    episode = original.model_copy(deep=True)
    fact = episode.oracle.facts[0]
    if change == "projection":
        episode.records[1].fields[fact.field_name] += increment
    elif change == "revision":
        episode.records[0].fields[fact.field_name] += increment
        fact.expected_value += increment
    elif change in ("supersession", "restored_precedence", "unobservable_override"):
        if change == "supersession":
            episode.records.insert(0, other)
        if change == "restored_precedence":
            episode.records.insert(0, episode.records.pop())
        fact.expected_value += increment
        fact.supporting_record_ids = [other.record_id]
    elif change == "unrelated_object":
        other.object_id += "-OTHER"
        episode.records.insert(0, other)
    elif change == "lower_priority_insert":
        episode.records.append(other)
    return original, episode


def run(seed, arm):
    rng = random.Random(seed)
    policy = PublicEvidenceReferencePolicy()
    rows = []
    for case in range(48):
        index = seed * 300 + case * 3 + case % 2
        for change in CHANGES:
            increment = rng.randint(1,500)
            original, episode = fixture_pair(index, increment, change)
            base = original.public_payload()
            payload = episode.public_payload()
            cached = policy(base)
            refreshed = fingerprint(base,cached,arm) != fingerprint(payload,cached,arm)
            fresh = policy(payload)
            answer = fresh if refreshed else cached
            rows.append(dict(index=index,increment=increment,change=change,refreshed=refreshed,
                             reward=verify_companyworld(answer,episode).overall_reward,
                             reference=verify_companyworld(fresh,episode).overall_reward,
                             submission=answer.model_dump(mode="json")))
    return rows


if __name__ == "__main__":
    rows = run(int(os.environ["TRAINLAB_SEED"]),sys.argv[1])
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    (output/"predictions.json").write_text(json.dumps(rows))
    score = sum(row["reward"]==1 for row in rows)/len(rows)
    (output/"metrics.json").write_text(json.dumps({"score":score}))
