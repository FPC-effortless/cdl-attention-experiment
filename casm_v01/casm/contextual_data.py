from __future__ import annotations

import random

import torch

from .explicit_compute import NUM_REGISTERS, VALUE_MODULUS, ProgramBatch, _apply_python

NUM_CONTEXT_COMMANDS = 4
CONTEXT_ALIASES = (2, 0, 3, 1)
CONTEXT_OPERATOR_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
HELDOUT_FAMILY_BIGRAMS = frozenset({(0, 1), (1, 2), (2, 3), (3, 0)})


def contextual_semantic(state: list[int], family: int, a: int, b: int, dst: int) -> int:
    context_bit = (state[a] ^ state[b] ^ state[dst]) & 1
    return CONTEXT_OPERATOR_PAIRS[family][context_bit]


def _sample_families(rng: random.Random, depth: int, split: str) -> list[int]:
    if split not in {"train", "iid", "composition"}:
        raise ValueError(split)
    for _ in range(10000):
        families = [rng.randrange(NUM_CONTEXT_COMMANDS) for _ in range(depth)]
        has_heldout = any(
            (left, right) in HELDOUT_FAMILY_BIGRAMS
            for left, right in zip(families, families[1:])
        )
        if split in {"train", "iid"} and not has_heldout:
            return families
        if split == "composition" and depth >= 2 and has_heldout:
            return families
    raise RuntimeError(f"could not sample {split} family sequence at depth={depth}")


def make_contextual_batch(batch_size: int, depth: int, seed: int, split: str = "train") -> ProgramBatch:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if split == "composition" and depth < 2:
        raise ValueError("composition split requires depth >= 2")
    rng = random.Random(seed)
    initials, commands, semantics, aa, bb, dd, targets = [], [], [], [], [], [], []
    for _ in range(batch_size):
        state = [rng.randrange(VALUE_MODULUS) for _ in range(NUM_REGISTERS)]
        initials.append(list(state))
        families = _sample_families(rng, depth, split)
        commands.append([CONTEXT_ALIASES[family] for family in families])
        row_semantics, row_a, row_b, row_d, row_targets = [], [], [], [], []
        for family in families:
            a, b, dst = [rng.randrange(NUM_REGISTERS) for _ in range(3)]
            semantic = contextual_semantic(state, family, a, b, dst)
            row_semantics.append(semantic)
            row_a.append(a)
            row_b.append(b)
            row_d.append(dst)
            state = _apply_python(state, semantic, a, b, dst)
            row_targets.append(list(state))
        semantics.append(row_semantics)
        aa.append(row_a)
        bb.append(row_b)
        dd.append(row_d)
        targets.append(row_targets)
    return ProgramBatch(
        initial=torch.tensor(initials, dtype=torch.long),
        commands=torch.tensor(commands, dtype=torch.long),
        semantics=torch.tensor(semantics, dtype=torch.long),
        arg_a=torch.tensor(aa, dtype=torch.long),
        arg_b=torch.tensor(bb, dtype=torch.long),
        dst=torch.tensor(dd, dtype=torch.long),
        target_states=torch.tensor(targets, dtype=torch.long),
    )
