from __future__ import annotations

from copy import deepcopy

import torch

from .explicit_compute import NUM_REGISTERS, VALUE_MODULUS, ProgramBatch
from .weak_supervision import SoftExplicitTransitionModel

REGIMES = ("full_final", "one_register", "one_parity_bit")


def cloned_partial_models(seed_model: SoftExplicitTransitionModel) -> dict[str, SoftExplicitTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    models = {}
    for regime in REGIMES:
        model = SoftExplicitTransitionModel(d_model=seed_model.d_model)
        model.load_state_dict(state)
        models[regime] = model
    return models


def sample_query_registers(batch_size: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    query = torch.randint(0, NUM_REGISTERS, (batch_size,), generator=generator)
    return query.to(device=device)


def _final_probs(model: SoftExplicitTransitionModel, batch: ProgramBatch) -> torch.Tensor:
    return model.rollout_soft(batch)[:, -1].clamp_min(1e-9)


def full_final_loss(model: SoftExplicitTransitionModel, batch: ProgramBatch) -> torch.Tensor:
    probs = _final_probs(model, batch)
    target = batch.target_states[:, -1]
    target_prob = probs.gather(2, target[:, :, None]).squeeze(-1)
    return -target_prob.log().mean()


def one_register_loss(
    model: SoftExplicitTransitionModel,
    batch: ProgramBatch,
    query_register: torch.Tensor,
) -> torch.Tensor:
    probs = _final_probs(model, batch)
    rows = torch.arange(probs.shape[0], device=probs.device)
    selected = probs[rows, query_register]
    target = batch.target_states[rows, -1, query_register]
    target_prob = selected.gather(1, target[:, None]).squeeze(1)
    return -target_prob.log().mean()


def one_parity_bit_loss(
    model: SoftExplicitTransitionModel,
    batch: ProgramBatch,
    query_register: torch.Tensor,
) -> torch.Tensor:
    probs = _final_probs(model, batch)
    rows = torch.arange(probs.shape[0], device=probs.device)
    selected = probs[rows, query_register]
    target_value = batch.target_states[rows, -1, query_register]
    target_parity = torch.remainder(target_value, 2)

    values = torch.arange(VALUE_MODULUS, device=probs.device)
    even_mask = torch.remainder(values, 2).eq(0).to(dtype=probs.dtype)
    odd_mask = 1.0 - even_mask
    parity_probs = torch.stack(
        [selected @ even_mask, selected @ odd_mask],
        dim=1,
    ).clamp_min(1e-9)
    target_prob = parity_probs.gather(1, target_parity[:, None]).squeeze(1)
    return -target_prob.log().mean()


def regime_loss(
    model: SoftExplicitTransitionModel,
    batch: ProgramBatch,
    regime: str,
    query_register: torch.Tensor,
) -> torch.Tensor:
    if regime == "full_final":
        return full_final_loss(model, batch)
    if regime == "one_register":
        return one_register_loss(model, batch, query_register)
    if regime == "one_parity_bit":
        return one_parity_bit_loss(model, batch, query_register)
    raise ValueError(regime)
