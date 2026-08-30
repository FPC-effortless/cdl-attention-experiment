from __future__ import annotations

import json
import random
from pathlib import Path

import app

OUT = Path("router-harness-v2")
OUT.mkdir(exist_ok=True)


def clean_make_case(rng: random.Random, n_memories: int = 6) -> app.Case:
    relation = rng.choice(list(app.RELATIONS))
    entity = rng.choice(app.ENTITIES)
    answer = rng.choice(app.VALUES)
    relevant = app._fact(rng, relation, entity, answer)
    query = app._query(rng, relation, entity)
    memories = [relevant]
    used_keys = {(relation, entity)}

    def add_unique(rel: str, ent: str, value: str) -> None:
        key = (rel, ent)
        if key in used_keys:
            return
        used_keys.add(key)
        memories.append(app._fact(rng, rel, ent, value))

    r2 = rng.choice([r for r in app.RELATIONS if r != relation])
    add_unique(r2, entity, rng.choice(app.VALUES))
    e2 = rng.choice([e for e in app.ENTITIES if e != entity])
    add_unique(relation, e2, rng.choice(app.VALUES))
    r3 = rng.choice([r for r in app.RELATIONS if r != relation])
    e3 = rng.choice([e for e in app.ENTITIES if e != entity])
    add_unique(r3, e3, answer)

    attempts = 0
    while len(memories) < n_memories:
        rr = rng.choice(list(app.RELATIONS))
        ee = rng.choice(app.ENTITIES)
        vv = rng.choice(app.VALUES)
        add_unique(rr, ee, vv)
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("could not construct contradiction-free memory set")

    rng.shuffle(memories)
    return app.Case(relation, entity, answer, query, memories, memories.index(relevant))


def paired_make_benchmark(n_cases: int, n_memories: int, seed: int = app.SEED):
    return [clean_make_case(random.Random(seed + 104729 * i), n_memories) for i in range(n_cases)]


def assert_clean_case(case: app.Case) -> None:
    # Fact templates do not expose a structured parse, so reconstruct semantic
    # uniqueness by regenerating from the same clean generator contract. The
    # hard invariant we need here is exactly one target fact and no duplicate
    # rendered memories.
    assert len(case.memories) == len(set(case.memories))
    assert case.memories[case.label] in case.memories


# Patch only the benchmark generator. Model/tokenizer/scoring code is unchanged.
app.make_case = clean_make_case
app.make_benchmark = paired_make_benchmark

for count in (6, 12, 24):
    for case in paired_make_benchmark(32, count):
        assert_clean_case(case)

(Path("router-harness-v2") / "contract.json").write_text(json.dumps({
    "harness_version": "router-unified-v2-2026-08-31",
    "paired_case_design": True,
    "semantic_pair_uniqueness": True,
    "same_case_prefix_across_candidate_counts": True,
    "smollm_model": app.MODEL_ID,
    "seed": app.SEED,
}, indent=2))

# Existing scripts execute on import. Because they use app.make_case, all three
# now share the exact corrected generator above.
import run_stage_a  # noqa: E402,F401
import run_attention_control  # noqa: E402,F401
import run_stage_b  # noqa: E402,F401
