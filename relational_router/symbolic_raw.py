"""Strong syntax-aware symbolic comparison for LRR-003.

Uses the very same training documents and *answer strings* as the neural
model. It receives a grammar-aware clause parser, but no family-ID labels.
"""
from __future__ import annotations
import random

from .raw_causal import tokenize, generate


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def root(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def unite(self, first: str, second: str) -> None:
        a, b = self.root(first), self.root(second)
        if a != b:
            self.parent[b] = a

    def unique_roots(self) -> int:
        return len({self.root(k) for k in self.parent})


def parse_raw(text: str) -> tuple[list[tuple[str, str, str, int]], str, str]:
    """Explicit three-word clause grammar: a deliberately strong baseline."""
    tokens = tokenize(text)
    if len(tokens) < 9 or tokens[-5:-4] != ("what",) or tokens[-1] != "?":
        raise ValueError("symbolic baseline knows only registered raw grammar")
    entity, predicate = tokens[-3], tokens[-2]
    facts: list[tuple[str, str, str, int]] = []
    pos = 0
    stop = len(tokens) - 5
    while pos < stop:
        if pos + 3 >= stop or tokens[pos+3] != ".":
            raise ValueError("unsupported clause syntax")
        facts.append((tokens[pos], tokens[pos+1], tokens[pos+2], pos+2))
        pos += 4
    return facts, entity, predicate


class SymbolicRawIndex:
    def __init__(self):
        self.families = UnionFind()
        self.training_documents = 0
        self.trainable_parameters = 0

    def observe_training(self, text: str, answer: str) -> None:
        facts, _, query_predicate = parse_raw(text)
        matching = [f for f in facts if f[2] == answer]
        if len(matching) != 1:
            raise ValueError("answer must uniquely identify training fact")
        self.families.unite(matching[0][1], query_predicate)
        self.training_documents += 1

    def train(self, seed: int, *, steps: int = 700, batch_size: int = 48) -> None:
        for step in range(steps):
            for i in range(batch_size):
                example = generate(
                    seed * 10_000_000 + step * 1_000 + i,
                    n_facts=random.Random(seed + step * 1_000 + i).randint(5, 8),
                    split="train")
                self.observe_training(example.text, example.answer)

    def route(self, text: str, *, relation_only: bool = False,
              literal_relation: bool = False) -> dict:
        facts, subject, predicate = parse_raw(text)
        relation = predicate if literal_relation else self.families.root(predicate)
        key = ("*" if relation_only else subject, relation)
        index: dict[tuple[str, str], int] = {}
        for entity, pred, _, token_index in facts:
            candidate = ("*" if relation_only else entity,
                         pred if literal_relation else self.families.root(pred))
            index[candidate] = token_index
        answer = index.get(key)
        words = tokenize(text)
        return {
            "token_index": answer,
            "token": words[answer] if answer is not None else None,
            "query_address": key,
            "admissions": len(facts),
            "distinct_addresses": len(index),
            "index_probes": 1,
        }
