"""LRR-004: test whether a soft local-address cost repairs wrong-offset basins.

The only trained intervention is relation-offset distance. Labels remain
answer-derived token positions, not predicate tags or canonical relation IDs.
OOV guard is separately marked as an engineered vocabulary support check.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .raw_causal import (
    ALIASES, NAMES, VOCAB, IDS, Example, CausalRoleIndexer,
    examples_to_batch, tokenize,
)


def make_example(seed: int, n_facts: int, split: str = "train") -> Example:
    """Matched new split: train omits (1,2),(3,0); dev (3,0); test (1,2)."""
    if n_facts < 5:
        raise ValueError("need at least five facts to construct hard distractors")
    if split not in ("train", "dev", "test"):
        raise ValueError("unknown split")
    rng = random.Random(seed)
    entity = rng.choice(NAMES)
    family = rng.choice(tuple(ALIASES))
    aliases = ALIASES[family]
    if split == "train":
        candidates = [
            (a, b) for i, a in enumerate(aliases)
            for j, b in enumerate(aliases)
            if (i, j) not in ((1, 2), (3, 0))
        ]
        fact_alias, query_alias = rng.choice(candidates)
    elif split == "dev":
        fact_alias, query_alias = aliases[3], aliases[0]
    else:
        fact_alias, query_alias = aliases[1], aliases[2]

    answer = f"value_{seed}_gold"
    facts = [(entity, fact_alias, answer, family)]
    for i, other_family in enumerate(x for x in ALIASES if x != family):
        facts.append((
            entity, rng.choice(ALIASES[other_family]),
            f"value_{seed}_wrongrel_{i}", other_family,
        ))
    wrong_entity = rng.choice([name for name in NAMES if name != entity])
    facts.append((
        wrong_entity, rng.choice(aliases),
        f"value_{seed}_wrongentity", family,
    ))
    while len(facts) < n_facts:
        other_family = rng.choice(tuple(ALIASES))
        name = rng.choice(NAMES)
        if name == entity and other_family == family:
            continue
        facts.append((
            name, rng.choice(ALIASES[other_family]),
            f"value_{seed}_distractor_{len(facts)}", other_family,
        ))
    rng.shuffle(facts)
    words: list[str] = []
    for subj, predicate, value, _ in facts:
        words.extend((subj, predicate, value, "."))
    words.extend(("what", "does", entity, query_alias, "?"))
    if words.count(answer) != 1:
        raise AssertionError("gold answer collision")
    index = words.index(answer)
    return Example(" ".join(words), answer, index, family, fact_alias, query_alias)


def locality_loss(
    model: CausalRoleIndexer,
    ids: torch.Tensor,
    answer: torch.Tensor,
    query: torch.Tensor,
    *,
    distance_weight: float = 0.0,
    balance_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single-forward exact source objective with one optional distance cost."""
    if not (0 <= distance_weight <= 1):
        raise ValueError("invalid distance weight")
    entity_logits, relation_logits, code_logits, gate, previous = model(ids)
    b, seq = ids.shape
    rows = torch.arange(b, device=ids.device)
    entity_p = entity_logits.softmax(-1)
    relation_p = relation_logits.softmax(-1)
    code_p = code_logits.softmax(-1)
    query_entity = entity_p[rows, query]
    query_window = previous[rows, query]
    equality = (
        (query_window[:, None, :, None] == previous[:, :, None, :]) &
        (query_window[:, None, :, None] != 0) &
        (previous[:, :, None, :] != 0)
    ).float()
    same_entity = (
        equality * query_entity[:, None, :, None] *
        entity_p[:, :, None, :]
    ).sum(dim=(-1, -2))
    read_relation = (
        relation_p[rows, query, :, None] * code_p[rows, query]
    ).sum(dim=1)
    write_relation = (relation_p[..., None] * code_p).sum(dim=2)
    same_relation = (
        write_relation * read_relation[:, None, :]
    ).sum(-1)
    match = (
        same_entity * same_relation * gate.sigmoid()
    ).clamp(1e-10, 1)
    causal = torch.arange(seq, device=ids.device)[None, :] < query[:, None]
    logits = torch.where(
        causal, match.log(), torch.full_like(match, -1e9),
    )
    answer_loss = F.cross_entropy(logits, answer)
    gold_relation = write_relation[rows, answer]
    usage = 0.0
    for distribution in (read_relation, gold_relation):
        marginal = distribution.mean(dim=0)
        usage = usage + (
            marginal *
            (marginal.clamp_min(1e-8) * model.codes).log()
        ).sum()
    offset_entropy = 0.0
    for distribution in (
        entity_p[rows, answer], entity_p[rows, query],
        relation_p[rows, answer], relation_p[rows, query],
    ):
        offset_entropy = offset_entropy - (
            distribution * distribution.clamp_min(1e-9).log()
        ).sum(-1).mean()
    gate_positive = F.softplus(-gate[rows, answer]).mean()
    gate_budget = (gate.sigmoid() * causal).sum() / causal.sum()

    lag = torch.arange(model.window, device=ids.device, dtype=relation_p.dtype)
    distance = 0.5 * (
        (relation_p[rows, answer] * lag).sum(-1).mean() +
        (relation_p[rows, query] * lag).sum(-1).mean()
    )
    loss = (
        answer_loss + 0.1 * gate_positive +
        0.03 * gate_budget + balance_weight * usage +
        0.005 * offset_entropy + distance_weight * distance
    )
    return loss, {
        "answer_loss": answer_loss.detach(),
        "distance": distance.detach(),
        "gate_budget": gate_budget.detach(),
        "usage": usage.detach(),
    }


def train_arm(
    seed: int, *, distance_weight: float, steps: int = 700,
    batch_size: int = 48,
) -> tuple[CausalRoleIndexer, dict, frozenset[str]]:
    """The streams and optimizer are seed-identical between matched arms."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CausalRoleIndexer(width=48, window=4, codes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    begin = time.monotonic()
    all_query_words: set[str] = set()
    lossval = 0.0
    distance = 0.0
    for step in range(steps):
        data = [
            make_example(
                seed * 10_000_000 + step * 1_000 + i,
                n_facts=random.Random(seed + step * 1_000 + i).randint(5, 8),
                split="train",
            )
            for i in range(batch_size)
        ]
        all_query_words.update(example.question_alias for example in data)
        ids, answer, query = examples_to_batch(data)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, obs = locality_loss(
            model, ids, answer, query, distance_weight=distance_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        lossval = float(loss.detach().item())
        distance = float(obs["distance"].item())
    model.eval()
    return model, {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "training_documents": steps * batch_size,
        "distance_weight": distance_weight,
        "final_loss": lossval,
        "final_mean_relation_offset": distance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_wall_seconds": time.monotonic() - begin,
        "observed_query_lexicon_size": len(all_query_words),
    }, frozenset(all_query_words)


class KnownLexiconGuard:
    """Explicit non-neural safety prior: known query lexical support only.

    Does not infer semantics or novel relations; it deliberately abstains on
    unfamiliar predicate words even if they are legitimate synonyms.
    """
    def __init__(
        self, model: CausalRoleIndexer,
        supported_query_words: frozenset[str],
    ):
        self.model = model
        self.supported_query_words = supported_query_words

    def route(self, text: str) -> dict:
        tokens = tokenize(text)
        if len(tokens) < 2 or tokens[-1] != "?":
            raise ValueError("question with '?' required")
        predicate = tokens[-2]
        if predicate not in self.supported_query_words:
            return {
                "token_index": None, "token": None,
                "query_address": None, "admissions": None,
                "distinct_addresses": None, "index_probes": 0,
                "abstained_reason": "predicate_not_in_training_lexicon",
            }
        result = self.model.route(text)
        return {**result, "abstained_reason": None}


def known_absent_challenge(example: Example) -> str:
    """Remove only the matching entity/family fact without answer leakage."""
    words = list(example.tokens)
    subject = words[-3]
    assert words[example.gold_index] == example.answer
    assert words[example.gold_index - 2] == subject
    replacement = next(name for name in NAMES if name != subject)
    words[example.gold_index - 2] = replacement
    for i in range(0, len(words) - 5, 4):
        if words[i] == subject and words[i + 1] in ALIASES[example.family]:
            raise AssertionError("known-absent control accidentally has a true fact")
    return " ".join(words)


def unknown_predicate_challenge(example: Example) -> str:
    words = list(example.tokens)
    words[-2] = "unseenverb"
    return " ".join(words)


def distant_predicate_challenge(example: Example) -> tuple[str, int]:
    """Push target fact predicate 1 step farther from answer (OOD syntax)."""
    words = list(example.tokens)
    words.insert(example.gold_index, "now")
    assert words[example.gold_index + 1] == example.answer
    return " ".join(words), example.gold_index + 1
