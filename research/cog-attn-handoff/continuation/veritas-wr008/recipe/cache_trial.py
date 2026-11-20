"""Public constructed CompanyWorld cache invalidation diagnostic."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from investigation_world.benchmark.policies import PublicEvidenceReferencePolicy
from investigation_world.companyworld import verifier
from research_lab.ecosystem import _fixture


def digest_citations(payload, answer):
    cited = {item["record_id"] for item in answer.evidence}
    records = [row for row in payload["records"] if row["record_id"] in cited]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def run(seed, guard):
    rng = random.Random(seed)
    policy = PublicEvidenceReferencePolicy()
    rows = []
    for case in range(48):
        index = seed * 300 + case * 3 + case % 2
        original = _fixture(index)
        initial = policy(original.public_payload())
        identity = digest_citations(original.public_payload(), initial)
        for change in ("unchanged", "projection", "revision", "supersession"):
            episode = original.model_copy(deep=True)
            fact = episode.oracle.facts[0]
            updated = fact.expected_value + rng.randint(1, 500)
            source, projection = episode.records
            if change == "projection":
                projection.fields[fact.field_name] = updated
            if change == "revision":
                source.fields[fact.field_name] = updated
                fact.expected_value = updated
            if change == "supersession":
                replacement = source.model_copy(deep=True)
                replacement.record_id += "-NEW"
                replacement.fields[fact.field_name] = updated
                episode.records.insert(0, replacement)
                fact.expected_value = updated
                fact.supporting_record_ids = [replacement.record_id]
            payload = episode.public_payload()
            refresh = guard and digest_citations(payload, initial) != identity
            answer = policy(payload) if refresh else initial
            score = verifier.verify_companyworld(answer, episode).overall_reward
            reference = verifier.verify_companyworld(policy(payload), episode).overall_reward
            assert reference == 1, (index, change, reference)
            rows.append(dict(index=index, change=change, refreshed=refresh, reward=score,
                             submission=answer.model_dump(mode="json")))
    return rows


if __name__ == "__main__":
    rows = run(int(os.environ["TRAINLAB_SEED"]), sys.argv[1] == "candidate")
    root = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    for name, value in [("predictions", rows), ("metrics", {"score":sum(r["reward"]==1 for r in rows)/len(rows)})]:
        (root / (name+".json")).write_text(json.dumps(value,sort_keys=True))
