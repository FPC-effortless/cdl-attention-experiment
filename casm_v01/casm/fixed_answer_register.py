from __future__ import annotations

from copy import deepcopy

import torch

from .explicit_compute import NUM_REGISTERS, ProgramBatch
from .partial_final_answer import full_final_loss, one_register_loss, sample_query_registers
from .weak_supervision import SoftExplicitTransitionModel

FIXED_REGISTER = 0
REGIMES = ("full_final", "random_register", "fixed_register")


def cloned_fixed_answer_models(
    seed_model: SoftExplicitTransitionModel,
) -> dict[str, SoftExplicitTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    models = {}
    for regime in REGIMES:
        model = SoftExplicitTransitionModel(d_model=seed_model.d_model)
        model.load_state_dict(state)
        models[regime] = model
    return models


def fixed_query_registers(
    batch_size: int,
    *,
    register: int = FIXED_REGISTER,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if not 0 <= register < NUM_REGISTERS:
        raise ValueError(register)
    return torch.full((batch_size,), int(register), dtype=torch.long, device=device)


def regime_loss(
    model: SoftExplicitTransitionModel,
    batch: ProgramBatch,
    regime: str,
    random_query: torch.Tensor,
) -> torch.Tensor:
    if regime == "full_final":
        return full_final_loss(model, batch)
    if regime == "random_register":
        return one_register_loss(model, batch, random_query)
    if regime == "fixed_register":
        fixed_query = fixed_query_registers(
            batch.initial.shape[0],
            device=batch.initial.device,
        )
        return one_register_loss(model, batch, fixed_query)
    raise ValueError(regime)


def sample_random_query_registers(
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return sample_query_registers(batch_size, seed, device=device)
