"""Attention-free causal relation addressing: an independently measurable pilot.

The model learns a *discrete relation code* for each predicate surface form.
Runtime token retrieval uses a Python posting-list lookup, never QK scoring,
a token-pair matrix, softmax aggregation, or exhaustive history ranking.

Deliberately scaffolded: entity extraction and fact/query boundaries are
provided. Training uses labeled predicate concepts, not answer-only credit.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


# Vocabulary / labels belong to the *training fixture*, never the index.
RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "ownership": ("owns", "possesses", "keeps", "holds"),
    "residence": ("lives", "resides", "dwells", "stays"),
    "preference": ("likes", "favors", "prefers", "enjoys"),
    "employment": ("employs", "hires", "contracts", "recruits"),
    "teaching": ("teaches", "tutors", "instructs", "coaches"),
    "discovery": ("found", "discovered", "uncovered", "detected"),
}


class LexicalRoleCoder(nn.Module):
    """Small learned concept encoder, independent of sequence length.

    Semantic labels are provided during training for this *first* pilot.
    Inference receives only the predicate token. No gold labels are supplied.
    """

    def __init__(self, vocabulary: tuple[str, ...], n_codes: int, width: int = 24):
        super().__init__()
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: i for i, token in enumerate(vocabulary)}
        if len(self.token_to_id) != len(vocabulary):
            raise ValueError("duplicate predicate tokens")
        if n_codes < 2 or width < 2:
            raise ValueError("invalid code count or width")
        self.embedding = nn.Embedding(len(vocabulary), width)
        self.projection = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, n_codes))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids))

    @torch.no_grad()
    def code(self, predicate: str) -> int | None:
        token_id = self.token_to_id.get(predicate.lower())
        if token_id is None:
            return None  # Unknown relation: abstain, never guess a bucket.
        dev = next(self.parameters()).device
        logits = self.forward(torch.tensor([token_id], device=dev))
        return int(logits.argmax(dim=-1).item())


def fit_coder(
    aliases: Mapping[str, tuple[str, ...]] = RELATION_ALIASES,
    *,
    seed: int = 0,
    steps: int = 120,
    width: int = 24,
    lr: float = 0.04,
) -> tuple[LexicalRoleCoder, dict[str, float]]:
    """Supervised lexical alignment only; does not train on evaluation stories."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    families = tuple(aliases)
    vocabulary = tuple(word for family in families for word in aliases[family])
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("relation aliases must be unambiguous")
    coder = LexicalRoleCoder(vocabulary, len(families), width)
    x = torch.arange(len(vocabulary))
    y = torch.tensor([i for i, family in enumerate(families) for _ in aliases[family]])
    optimizer = torch.optim.AdamW(coder.parameters(), lr=lr, weight_decay=0.001)
    coder.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(coder(x), y)
        loss.backward()
        optimizer.step()
    coder.eval()
    with torch.no_grad():
        accuracy = (coder(x).argmax(dim=-1) == y).float().mean().item()
    return coder, {"train_lexical_accuracy": float(accuracy), "optimizer_steps": steps}


@dataclass(frozen=True)
class AddressedToken:
    token_index: int
    entity: str
    relation_code: int
    value: str


@dataclass(frozen=True)
class Selection:
    indices: tuple[int, ...]
    values: tuple[str, ...]
    posting_visits: int
    index_probes: int


class CausalRelationIndex:
    """Incremental hash index; lookup cost does not scan unrelated facts.

    token_index is the absolute position of the value token in a token stream.
    Call observe in strictly increasing token order; query at the next position.
    Selection returns positions, *not* a weighted sum of token vectors.
    """

    def __init__(self, coder: LexicalRoleCoder):
        self.coder = coder
        self._postings: dict[tuple[str, int], list[AddressedToken]] = defaultdict(list)
        self._last_observed_index = -1
        self.observed_facts = 0

    def observe(self, *, token_index: int, entity: str, predicate: str, value: str) -> None:
        if token_index <= self._last_observed_index:
            raise ValueError("facts must arrive in strictly increasing causal order")
        if token_index < 0 or not entity:
            raise ValueError("invalid token index or entity")
        self._last_observed_index = token_index
        self.observed_facts += 1
        code = self.coder.code(predicate)
        if code is not None:
            key = (entity, code)
            self._postings[key].append(AddressedToken(token_index, entity, code, value))

    def select(self, *, query_index: int, entity: str, predicate: str, k: int = 1) -> Selection:
        if query_index <= self._last_observed_index:
            raise ValueError("query must be strictly after all observed facts")
        if k < 1:
            raise ValueError("retrieval budget k must be positive")
        code = self.coder.code(predicate)
        if code is None:
            return Selection((), (), 0, 0)
        selected = self._postings.get((entity, code), [])[-k:]
        # A defensive causality assertion; it should follow from observe order.
        if any(item.token_index >= query_index for item in selected):
            raise AssertionError("future-token leakage")
        newest_first = selected[::-1]
        return Selection(
            indices=tuple(item.token_index for item in newest_first),
            values=tuple(item.value for item in newest_first),
            posting_visits=len(newest_first),
            index_probes=1,
        )

    @property
    def distinct_addresses(self) -> int:
        return len(self._postings)
