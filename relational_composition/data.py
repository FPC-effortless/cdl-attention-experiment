"""Controlled raw text; semantic metadata is evaluation-only.

This is a mechanism dataset, NOT a natural-language corpus benchmark. Relations
are random functions, so reversing their order usually changes the answer. Names
and withheld compositions/paraphrases are separate diagnostic axes.
"""
from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class Example:
    text: str
    answer_start: int  # byte offset; never supplied to model.forward
    answer: str
    fact_ends: tuple[int, int]


def make_example(seed: int, split="train", condition="iid", entities=4):
    if split not in ("train", "dev", "test"):
        raise ValueError("unknown split")
    if condition not in ("iid", "names", "composition", "paraphrase", "updates", "older"):
        raise ValueError("unknown diagnostic")
    if entities < 3 or entities > 20:
        raise ValueError("entities must be in [3, 20]")
    if split == "train" and condition != "iid":
        raise ValueError("diagnostic populations must not enter training")
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrst" if condition != "names" else "uvwxyz0123456789ABCDE"
    names = rng.sample(["n" + a for a in alphabet], entities)
    rels = ("follows", "visits")
    maps = {r: dict(zip(names, rng.sample(names, len(names)))) for r in rels}
    pair = ("visits", "follows") if condition == "composition" else rng.choice(
        [("follows", "visits"), ("follows", "follows"), ("visits", "visits")])
    start = rng.choice(names)
    facts = [(name, r, maps[r][name]) for name in names for r in rels]
    rng.shuffle(facts)
    text, ends = "", {}
    for name, rel, value in facts:
        text += f"{name} {rel} {value}. "
        ends[(name, rel)] = len(text.encode()) - 2
    if condition in ("updates", "older"):
        # Explicit wording resolves version semantics; no silent contradictory gold.
        old_maps = {r: dict(m) for r, m in maps.items()}
        changed = rng.choice([x for x in names if x != maps[pair[0]][start]])
        text += f"Now {start} {pair[0]} {changed}. "
        if condition == "updates":
            maps[pair[0]][start] = changed
            ends[(start, pair[0])] = len(text.encode()) - 2
        else:
            maps = old_maps
    middle = maps[pair[0]][start]
    answer = maps[pair[1]][middle]
    if condition == "paraphrase":
        question = f"From {start}, take {pair[0]} then {pair[1]}. Destination? "
    else:
        prefix = "Before the update, " if condition == "older" else ""
        question = f"{prefix}Where from {start} via {pair[0]} then {pair[1]}? "
    text += question
    return Example(text + answer + "\n", len(text.encode()), answer,
                   (ends[(start, pair[0])], ends[(middle, pair[1])]))


def encode(text: str):
    return [byte + 1 for byte in text.encode("utf-8")]


def batch(examples):
    encoded = [encode(e.text) for e in examples]
    length = max(map(len, encoded)) - 1
    x = torch.zeros(len(encoded), length, dtype=torch.long)
    y = torch.zeros_like(x)
    answer_mask = torch.zeros_like(x, dtype=torch.bool)
    for row, (values, example) in enumerate(zip(encoded, examples)):
        x[row, :len(values) - 1] = torch.tensor(values[:-1])
        y[row, :len(values) - 1] = torch.tensor(values[1:])
        start = example.answer_start - 1
        answer_mask[row, start:start + len(example.answer.encode())] = True
    return x, y, answer_mask
