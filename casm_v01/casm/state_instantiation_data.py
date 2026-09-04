from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .contextual_data import CONTEXT_ALIASES
from .explicit_compute import VALUE_MODULUS, ProgramBatch, _apply_python
from .variable_contextual_data import contextual_semantic_variable

NUM_CANDIDATES = 8
TRAIN_LIVE_CARDINALITIES = (2, 3, 4)
UNSEEN_LIVE_CARDINALITIES = (5, 6)
OUTPUT_CANDIDATE = 0
# Family 1 maps to sub/max. In the causal suffix we make every non-output
# live value even and use a=acc,b=x,dst=acc, so the context bit is parity(x)=0
# and the semantic operator is always binary subtraction. This guarantees that
# every syntactic source on the certified live slice is also semantically used.
CAUSAL_FAMILY = 1


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
        ever_live.add(d)
        current_versions.remove(d)
        current_versions.add(a)
        current_versions.add(b)
        ever_live.add(a)
        ever_live.add(b)
    return [i in ever_live for i in range(NUM_CANDIDATES)]


def _construct_program(
    rng: random.Random,
    *,
    depth: int,
    target_live_cardinality: int,
    split: str,
) -> tuple[list[int], list[int], list[int], list[int], list[bool], list[int]]:
    """Construct, rather than rejection-sample, an exact temporal live/dead program.

    The final `n-1` operations form a multi-hop causal chain over a sampled live set.
    Earlier filler operations write only distractors, so they are syntactically active
    but cannot reach final candidate 0. The returned `live_mask` is independently
    recomputed by `backward_live_mask` and checked against the chosen set.
    """
    if split not in {"train", "iid", "composition"}:
        raise ValueError(split)
    if target_live_cardinality not in (*TRAIN_LIVE_CARDINALITIES, *UNSEEN_LIVE_CARDINALITIES):
        raise ValueError(target_live_cardinality)
    causal_steps = target_live_cardinality - 1
    if depth <= causal_steps:
        raise ValueError((depth, target_live_cardinality))

    live_nonzero = rng.sample(range(1, NUM_CANDIDATES), target_live_cardinality - 1)
    rng.shuffle(live_nonzero)
    live_set = {OUTPUT_CANDIDATE, *live_nonzero}
    dead = [i for i in range(1, NUM_CANDIDATES) if i not in live_set]
    if not dead:
        raise RuntimeError("X20 requires distractor candidates")

    filler_steps = depth - causal_steps
    aa: list[int] = []
    bb: list[int] = []
    dd: list[int] = []
    families: list[int] = []

    # Dead filler: cycle destinations through every distractor so mention count cannot
    # identify the live set. Sources mix live/dead candidates, but the destination is
    # never on the output slice and none of these values is consumed by the causal suffix.
    for t in range(filler_steps):
        d = dead[t % len(dead)]
        a = rng.randrange(NUM_CANDIDATES)
        b = rng.randrange(NUM_CANDIDATES)
        aa.append(a)
        bb.append(b)
        dd.append(d)
        families.append(CAUSAL_FAMILY)

    # Multi-hop live chain. Accumulate all non-output live candidates into one live
    # record, then fold that accumulator into output 0. Only the accumulator is directly
    # referenced by the final output write; the other live identities are upstream.
    accumulator = live_nonzero[0]
    for x in live_nonzero[1:]:
        aa.append(accumulator)
        bb.append(x)
        dd.append(accumulator)
        families.append(CAUSAL_FAMILY)
    aa.append(OUTPUT_CANDIDATE)
    bb.append(accumulator)
    dd.append(OUTPUT_CANDIDATE)
    families.append(CAUSAL_FAMILY)

    # The composition split differs only by inserting one held-out family transition in
    # dead code. This keeps the state-selection causal slice fixed while preserving the
    # existing suite label as an executor stress diagnostic rather than a constructor cue.
    if split == "composition" and filler_steps >= 2:
        families[0], families[1] = 0, 1

    derived = backward_live_mask(aa, bb, dd)
    expected = [i in live_set for i in range(NUM_CANDIDATES)]
    if derived != expected:
        raise RuntimeError((derived, expected, aa, bb, dd))
    if not any((not derived[i]) and i in (set(aa) | set(bb) | set(dd)) for i in range(NUM_CANDIDATES)):
        raise RuntimeError("constructed program lacks distractor mention")
    return aa, bb, dd, families, derived, live_nonzero


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

    for _ in range(batch_size):
        aa, bb, dd, families, live_mask, live_nonzero = _construct_program(
            rng, depth=depth, target_live_cardinality=live_cardinality, split=split
        )
        initial = [rng.randrange(VALUE_MODULUS) for _ in range(NUM_CANDIDATES)]
        # Guarantee binary subtraction on every operation of the causal suffix:
        # for a=acc,b=x,dst=acc the contextual bit reduces to parity(x).
        for i in live_nonzero:
            initial[i] = 2 * rng.randrange(VALUE_MODULUS // 2)

        state = list(initial)
        row_semantics, row_targets = [], []
        for family, a, b, d in zip(families, aa, bb, dd):
            semantic = contextual_semantic_variable(state, family, a, b, d)
            row_semantics.append(semantic)
            state = _apply_python(state, semantic, a, b, d)
            row_targets.append(list(state))

        # Every certified causal-suffix operation must actually be binary subtraction.
        for semantic in row_semantics[-(live_cardinality - 1):]:
            if semantic != 2:
                raise RuntimeError(f"causal suffix lost binary-dependency guarantee: {semantic}")

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
