"""Reproducible controlled retrieval benchmark for relation-addressed tokens.

No prompt language model, no gold-position input to the indexed router,
no query-to-history comparisons in the learned runtime. Each document uses
an explicit FACT/entity/predicate/value schema: this is not unrestricted NLP.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from .router import CausalRelationIndex, RELATION_ALIASES, LexicalRoleCoder, fit_coder


@dataclass(frozen=True)
class Fact:
    entity: str
    predicate: str
    value: str
    family: str
    token_index: int = -1


@dataclass(frozen=True)
class Trial:
    facts: tuple[Fact, ...]
    tokens: tuple[str, ...]
    query_entity: str
    query_predicate: str
    query_index: int
    gold_value: str
    gold_token_index: int


def build_trial(n_facts: int, seed: int) -> Trial:
    """Fresh opaque entity/value symbols, hard negatives, shuffled order.

    Every trial includes one target fact, same-entity/wrong-relation
    distractors, and same-relation/wrong-entity distractors. Query alias is
    deliberately different from the target's lexical predicate.
    """
    if n_facts < 2:
        raise ValueError("at least two facts are needed")
    rng = random.Random(seed)
    families = tuple(RELATION_ALIASES)
    target_family = rng.choice(families)
    target_entity = f"entity_{seed}_target"
    target_alias, query_alias = rng.sample(RELATION_ALIASES[target_family], 2)
    facts = [Fact(target_entity, target_alias, f"value_{seed}_gold", target_family)]
    wrong_families = [family for family in families if family != target_family]
    rng.shuffle(wrong_families)
    for family in wrong_families[: min(n_facts - 1, len(wrong_families))]:
        facts.append(Fact(target_entity, rng.choice(RELATION_ALIASES[family]),
                          f"value_{seed}_wrong_role_{family}", family))
    while len(facts) < n_facts:
        i = len(facts)
        family = target_family if i % 2 else rng.choice(families)
        facts.append(Fact(f"entity_{seed}_{i}", rng.choice(RELATION_ALIASES[family]),
                          f"value_{seed}_decoy_{i}", family))
    rng.shuffle(facts)
    tokens: list[str] = []
    placed: list[Fact] = []
    for fact in facts:
        start = len(tokens)
        tokens.extend(("<F>", fact.entity, fact.predicate, fact.value, "</F>"))
        placed.append(Fact(fact.entity, fact.predicate, fact.value, fact.family, start + 3))
    target = next(f for f in placed if f.entity == target_entity and f.family == target_family)
    # Query is appended *after* facts and is never inserted into the index.
    query_index = len(tokens)
    tokens.extend(("<Q>", target_entity, query_alias, "?", "</Q>"))
    return Trial(tuple(placed), tuple(tokens), target_entity, query_alias,
                 query_index, target.value, target.token_index)


def select_trial(coder: LexicalRoleCoder, trial: Trial, k: int = 1):
    index = CausalRelationIndex(coder)
    for fact in trial.facts:
        assert trial.tokens[fact.token_index] == fact.value
        index.observe(token_index=fact.token_index, entity=fact.entity,
                      predicate=fact.predicate, value=fact.value)
    started = time.perf_counter_ns()
    selection = index.select(query_index=trial.query_index,
                             entity=trial.query_entity,
                             predicate=trial.query_predicate, k=k)
    query_us = (time.perf_counter_ns() - started) / 1000
    return selection, index, query_us


def benchmark(
    *,
    lengths: tuple[int, ...] = (8, 64, 256, 1024),
    trials: int = 64,
    seed: int = 17,
    steps: int = 120,
    k: int = 1,
) -> dict:
    if trials < 1 or not lengths or k < 1:
        raise ValueError("trials, lengths and k must be positive")
    coder, training = fit_coder(seed=seed, steps=steps)
    result: dict = {
        "method": "causal_hash_relation_router_v0.1",
        "training": training,
        "constraints": {
            "entity_extraction": "supplied",
            "fact_segmentation": "supplied",
            "relation_alignment": "supervised_alias_labels",
            "no_token_pair_attention": True,
            "no_exhaustive_query_scan": True,
            "index_ingest_cost": "one write per fact; O(H) total",
            "query_cost": "O(1) address probe + O(k) returned postings",
            "k": k,
        },
        "rows": [],
    }
    for n_facts in lengths:
        success = surface = relation_only = oracle = 0
        probes = visits = 0
        runtimes_us: list[float] = []
        for trial_number in range(trials):
            trial = build_trial(n_facts, seed * 1000000 + n_facts * 1000 + trial_number)
            selected, index, elapsed = select_trial(coder, trial, k=k)
            success += int(trial.gold_token_index in selected.indices)
            # Exact predicate-token + entity baseline: no learned alias alignment.
            same_surface = [fact for fact in trial.facts
                            if fact.entity == trial.query_entity
                            and fact.predicate == trial.query_predicate]
            surface += int(any(f.token_index == trial.gold_token_index for f in same_surface[-k:]))
            # Structure-blind negative control: ignore entity, select newest
            # fact with the decoded relation (the baseline scans all facts).
            qcode = coder.code(trial.query_predicate)
            matching_role = [f for f in trial.facts if coder.code(f.predicate) == qcode]
            relation_only += int(any(f.token_index == trial.gold_token_index
                                     for f in matching_role[-k:]))
            # Oracle evaluation is entirely outside the learned index.
            gold = [f for f in trial.facts
                    if f.entity == trial.query_entity
                    and f.family == next(f.family for f in trial.facts
                                         if f.token_index == trial.gold_token_index)]
            oracle += int(any(f.token_index == trial.gold_token_index for f in gold[-k:]))
            probes += selected.index_probes
            visits += selected.posting_visits
            runtimes_us.append(elapsed)
            assert index.observed_facts == n_facts
            assert not selected.indices or max(selected.indices) < trial.query_index
        ordered_us = sorted(runtimes_us)
        result["rows"].append({
            "facts": n_facts,
            "token_context_length": len(trial.tokens),
            "trials": trials,
            "learned_recall_at_k": success / trials,
            "raw_predicate_recall_at_k": surface / trials,
            "relation_only_recall_at_k": relation_only / trials,
            "oracle_recall_at_k": oracle / trials,
            "mean_index_probes": probes / trials,
            "mean_postings_visited": visits / trials,
            "mean_ingest_writes": n_facts,
            "query_latency_median_us": float(statistics.median(runtimes_us)),
            "query_latency_p95_us": float(ordered_us[max(0, (95 * trials + 99) // 100 - 1)]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-free causal relation index pilot")
    parser.add_argument("--lengths", default="8,64,256,1024")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lengths = tuple(int(part) for part in args.lengths.split(","))
    report = benchmark(lengths=lengths, trials=args.trials, seed=args.seed,
                       steps=args.steps, k=args.k)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
