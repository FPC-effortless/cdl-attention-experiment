"""LRR-002: induce discrete runtime address codes from pairwise answer outcomes.

No canonical relation IDs are fed to the learner. A training pair only says
whether a query predicate can retrieve the value of a fact predicate for the
same entity. This is *not* open-world semantic discovery: all 24 lexemes are
seen in these training pairs, the number of codes is supplied, and the
supervision pairs were generated from a controlled relation fixture.

Training uses per-token categorical softmax and a pairwise agreement loss.
Neither query-time retrieval nor the causal index uses token-pair attention.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .router import LexicalRoleCoder, RELATION_ALIASES


@dataclass(frozen=True)
class PairOutcome:
    fact_predicate: str
    question_predicate: str
    answered: bool


def controlled_pair_outcomes(
    aliases: Mapping[str, tuple[str, ...]] = RELATION_ALIASES,
) -> tuple[tuple[str, ...], tuple[PairOutcome, ...]]:
    """Construct balanced supervision *about answerability*, not code labels.

    The fixture's relation families are private to data generation. An
    arbitrary permutation of code IDs leaves the task objective unchanged.
    """
    vocabulary = tuple(word for words in aliases.values() for word in words)
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("duplicate relation lexeme across fixture families")
    outcome_pairs: list[PairOutcome] = []
    entries = [(word, family) for family, words in aliases.items() for word in words]
    for i, (fact_predicate, fact_family) in enumerate(entries):
        for question_predicate, question_family in entries[i + 1 :]:
            outcome_pairs.append(PairOutcome(
                fact_predicate=fact_predicate,
                question_predicate=question_predicate,
                answered=fact_family == question_family,
            ))
    return vocabulary, tuple(outcome_pairs)


def fit_from_outcomes(
    vocabulary: Sequence[str],
    pair_outcomes: Sequence[PairOutcome],
    *,
    n_codes: int,
    seed: int = 0,
    steps: int = 800,
    width: int = 24,
    lr: float = 0.04,
    balance_weight: float = 0.5,
) -> tuple[LexicalRoleCoder, dict[str, float]]:
    """Learn addresses from answerability pairs without category-ID labels.

    Differentiable training-only P(same code) is computed from the two
    independent categorical assignments. The balance penalty prevents
    semantically different families collapsing into the same bucket.
    """
    if steps < 1 or n_codes < 2 or balance_weight < 0:
        raise ValueError("invalid steps, n_codes or balance_weight")
    vocabulary = tuple(vocabulary)
    torch.manual_seed(seed)
    coder = LexicalRoleCoder(vocabulary, n_codes=n_codes, width=width)
    words = torch.arange(len(vocabulary), dtype=torch.long)
    ix = coder.token_to_id
    facts = torch.tensor([ix[p.fact_predicate] for p in pair_outcomes], dtype=torch.long)
    questions = torch.tensor([ix[p.question_predicate] for p in pair_outcomes], dtype=torch.long)
    outcomes = torch.tensor([p.answered for p in pair_outcomes], dtype=torch.bool)
    if not outcomes.any() or not (~outcomes).any():
        raise ValueError("positive and negative answerability examples are required")
    optimizer = torch.optim.AdamW(coder.parameters(), lr=lr, weight_decay=0.001)

    coder.train()
    for _ in range(steps):
        # Categorical assignments for individual lexical codes; the tokens
        # belong to a small training lexicon, NOT to the full input sequence.
        probabilities = F.softmax(coder(words), dim=-1)
        same_code_prob = (probabilities[facts] * probabilities[questions]).sum(dim=-1)
        same_code_prob = same_code_prob.clamp(1e-8, 1 - 1e-8)
        positive_loss = -torch.log(same_code_prob[outcomes]).mean()
        negative_loss = -torch.log1p(-same_code_prob[~outcomes]).mean()
        mean_usage = probabilities.mean(dim=0)
        usage_kl = (mean_usage * torch.log((mean_usage * n_codes).clamp_min(1e-8))).sum()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(-1).mean()
        loss = positive_loss + negative_loss + balance_weight * usage_kl + 0.02 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    coder.eval()
    with torch.no_grad():
        codes = coder(words).argmax(-1)
        matched = codes[facts] == codes[questions]
        positive_acc = matched[outcomes].float().mean().item()
        negative_acc = (~matched[~outcomes]).float().mean().item()
        unique_codes = int(codes.unique().numel())
        marginal = torch.bincount(codes, minlength=n_codes).float() / len(codes)
        report = {
            "positive_pair_accuracy": float(positive_acc),
            "negative_pair_accuracy": float(negative_acc),
            "balanced_pair_accuracy": float((positive_acc + negative_acc) / 2),
            "unique_hard_codes": unique_codes,
            "hard_code_occupancy_entropy": float(
                -(marginal * marginal.clamp_min(1e-9).log()).sum().item()),
            "training_pair_count": int(len(pair_outcomes)),
            "positive_training_pairs": int(outcomes.sum().item()),
            "optimizer_steps": steps,
            "balance_weight": balance_weight,
            "parameter_count": sum(p.numel() for p in coder.parameters()),
        }
    return coder, report
