from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .contextual_data import CONTEXT_ALIASES
from .explicit_compute import VALUE_MODULUS, ProgramBatch, _apply_python
from .variable_contextual_data import _sample_families, contextual_semantic_variable

NUM_CANDIDATES = 8
TRAIN_LIVE_CARDINALITIES = (2, 3, 4)
UNSEEN_LIVE_CARDINALITIES = (5, 6)
OUTPUT_CANDIDATE = 0


@dataclass(frozen=True)
class StateInstantiationBatch:
    program: ProgramBatch
    live_mask: torch.Tensor  # evaluation/integrity only; never an input to learned constructor

    @property
    def batch_size(self) -> int:
        return int(self.program.initial.shape[0])


def backward_live_mask(arg_a: list[int], arg_b: list[int], dst: list[int]) -> list[bool]:
    """Return candidate identities with at least one version on the final-output dataflow slice.

    `current_versions` tracks which candidate versions immediately before the currently
    scanned operation are required by the final output. `ever_live` separately records
    candidate identities for which any version is on that slice. Keeping these sets
    separate is essential under destructive writes: an older overwritten version of a
    destination must not remain live merely because a later version of that candidate
    was causal.
    """
    if not (len(arg_a) == len(arg_b) == len(dst)):
        raise ValueError("program field lengths differ")
    current_versions = {OUTPUT_CANDIDATE}
    ever_live = {OUTPUT_CANDIDATE}
    for a, b, d in reversed(list(zip(arg_a, arg_b, dst))):
        a, b, d = int(a), int(b), int(d)
        if d not in current_versions:
            continue
        # The post-operation version of d is causal. Moving across this write replaces
        # that version with exactly the source versions consumed by the operation.
        ever_live.add(d)
        current_versions.remove(d)
        current_versions.add(a)
        current_versions.add(b)
        ever_live.add(a)
        ever_live.add(b)
    return [i in ever_live for i in range(NUM_CANDIDATES)]


def _program_mentions_distractor(
    live_mask: list[bool], arg_a: list[int], arg_b: list[int], dst: list[int]
) -> bool:
    mentioned = set(arg_a) | set(arg_b) | set(dst)
    return any((not live_mask[i]) and i in mentioned for i in range(NUM_CANDIDATES))


def _sample_operand_program(
    rng: random.Random,
    *,
    depth: int,
    target_live_cardinality: int,
) -> tuple[list[int], list[int], list[int], list[bool]]:
    if target_live_cardinality not in (*TRAIN_LIVE_CARDINALITIES, *UNSEEN_LIVE_CARDINALITIES):
        raise ValueError(target_live_cardinality)
    for _ in range(200_000):
        aa = [rng.randrange(NUM_CANDIDATES) for _ in range(depth)]
        bb = [rng.randrange(NUM_CANDIDATES) for _ in range(depth)]
        dd = [rng.randrange(NUM_CANDIDATES) for _ in range(depth)]
        live_mask = backward_live_mask(aa, bb, dd)
        if sum(live_mask) != target_live_cardinality:
            continue
        if not live_mask[OUTPUT_CANDIDATE]:
            continue
        if not _program_mentions_distractor(live_mask, aa, bb, dd):
            continue
        # Avoid degenerate samples where all live/distractor mentions are isolated to one field.
        live_mentions = sum(live_mask[x] for x in aa + bb + dd)
        dead_mentions = 3 * depth - live_mentions
        if live_mentions < depth or dead_mentions < max(1, depth // 4):
            continue
        return aa, bb, dd, live_mask
    raise RuntimeError(
        f"could not sample depth={depth} program with live cardinality {target_live_cardinality}"
    )


def make_state_instantiation_batch(
    batch_size: int,
    depth: int,
    seed: int,
    *,
    live_cardinality: int,
    split: str = "train",
) -> StateInstantiationBatch:
    if batch_size < 1 or depth < 1:
        raise ValueError((batch_size, depth))
    if live_cardinality not in (*TRAIN_LIVE_CARDINALITIES, *UNSEEN_LIVE_CARDINALITIES):
        raise ValueError(live_cardinality)

    rng = random.Random(seed)
    initials, commands, semantics = [], [], []
    aa_all, bb_all, dd_all, targets, masks = [], [], [], [], []

    for row in range(batch_size):
        initial = [rng.randrange(VALUE_MODULUS) for _ in range(NUM_CANDIDATES)]
        families = _sample_families(rng, depth, split)
        aa, bb, dd, live_mask = _sample_operand_program(
            rng, depth=depth, target_live_cardinality=live_cardinality
        )

        state = list(initial)
        row_semantics, row_targets = [], []
        for family, a, b, d in zip(families, aa, bb, dd):
            semantic = contextual_semantic_variable(state, family, a, b, d)
            row_semantics.append(semantic)
            state = _apply_python(state, semantic, a, b, d)
            row_targets.append(list(state))

        initials.append(initial)
        commands.append([CONTEXT_ALIASES[f] for f in families])
        semantics.append(row_semantics)
        aa_all.append(aa)
        bb_all.append(bb)
        dd_all.append(dd)
        targets.append(row_targets)
        masks.append(live_mask)

    return StateInstantiationBatch(
        program=ProgramBatch(
            initial=torch.tensor(initials, dtype=torch.long),
            commands=torch.tensor(commands, dtype=torch.long),
            semantics=torch.tensor(semantics, dtype=torch.long),
            arg_a=torch.tensor(aa_all, dtype=torch.long),
            arg_b=torch.tensor(bb_all, dtype=torch.long),
            dst=torch.tensor(dd_all, dtype=torch.long),
            target_states=torch.tensor(targets, dtype=torch.long),
        ),
        live_mask=torch.tensor(masks, dtype=torch.bool),
    )


def training_live_cardinality_for_step(step: int) -> int:
    if step < 1:
        raise ValueError(step)
    return TRAIN_LIVE_CARDINALITIES[(step - 1) % len(TRAIN_LIVE_CARDINALITIES)]
