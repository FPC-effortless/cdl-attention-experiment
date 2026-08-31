from __future__ import annotations

import math

import torch

from casm.run_stable_cardinality_executor import (
    GRAD_CLIP_NORM,
    LR_MAX,
    LR_MIN,
    PREREGISTERED_STEPS,
    REGIMES,
    WEIGHT_DECAY,
    cloned_local_models,
    cosine_decay_lr,
    paired_train_step,
)
from casm.variable_contextual_data import make_variable_contextual_batch


def test_cosine_schedule_exact_endpoints_and_monotonicity():
    assert cosine_decay_lr(1) == LR_MAX
    assert cosine_decay_lr(PREREGISTERED_STEPS) == LR_MIN
    lrs = [cosine_decay_lr(step) for step in range(1, PREREGISTERED_STEPS + 1)]
    assert all(right <= left for left, right in zip(lrs, lrs[1:]))


def test_cosine_schedule_matches_frozen_formula():
    for step in (1, 2, 137, 5000, 7777, 9999, 10000):
        expected = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (
            1.0 + math.cos(math.pi * (step - 1) / (PREREGISTERED_STEPS - 1))
        )
        assert math.isclose(cosine_decay_lr(step), expected, rel_tol=0.0, abs_tol=1e-15)


def test_paired_models_start_bit_identical_and_parameter_matched():
    torch.manual_seed(9001)
    models = cloned_local_models(d_model=32)
    assert tuple(models) == REGIMES
    first = models[REGIMES[0]].state_dict()
    second = models[REGIMES[1]].state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert models[REGIMES[0]].parameter_count() == models[REGIMES[1]].parameter_count()
    assert models[REGIMES[0]].trainable_parameter_count() == models[REGIMES[1]].trainable_parameter_count()


def test_paired_step_uses_same_batch_and_only_lr_differs():
    torch.manual_seed(9002)
    models = cloned_local_models(d_model=32)
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        for name, model in models.items()
    }
    batch = make_variable_contextual_batch(8, 8, 99002, num_registers=4, split="train")
    before = {
        name: {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        for name, model in models.items()
    }
    result = paired_train_step(models, optimizers, batch, step=5000)
    assert result["fixed_lr_replication"]["lr"] == LR_MAX
    assert result["cosine_decay_stable"]["lr"] == cosine_decay_lr(5000)
    assert result["cosine_decay_stable"]["lr"] < LR_MAX
    assert math.isfinite(result["fixed_lr_replication"]["loss"])
    assert math.isfinite(result["cosine_decay_stable"]["loss"])
    assert math.isfinite(result["fixed_lr_replication"]["grad_norm"])
    assert math.isfinite(result["cosine_decay_stable"]["grad_norm"])
    # Because the cloned models and batch are identical before the step, the pre-update
    # objective and gradient norm must match; only the optimizer learning rate may differ.
    assert math.isclose(
        result["fixed_lr_replication"]["loss"],
        result["cosine_decay_stable"]["loss"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        result["fixed_lr_replication"]["grad_norm"],
        result["cosine_decay_stable"]["grad_norm"],
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert all(torch.equal(before[REGIMES[0]][k], before[REGIMES[1]][k]) for k in before[REGIMES[0]])


def test_optimizer_constants_remain_preregistered():
    assert WEIGHT_DECAY == 1e-4
    assert GRAD_CLIP_NORM == 1.0
    assert LR_MAX == 2e-3
    assert LR_MIN == 2e-4
    assert PREREGISTERED_STEPS == 10000
