from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_REGISTERS, VALUE_MODULUS, ProgramBatch, StateCommandEncoder


class StateAccessGRUControl(nn.Module):
    """Recurrent control given the same current-state access as the explicit transition model.

    During training it receives the true previous state, exactly like SharedTransitionModel.
    During rollout it receives its own previously predicted state. A persistent GRU hidden state is
    retained in addition to that explicit state, so this is a conservative control: it has at least
    as much recurrent memory as the explicit transition MLP.
    """

    def __init__(self, d_model: int = 112):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.cell = nn.GRUCell(2 * d_model, d_model)
        self.head = nn.Linear(d_model, VALUE_MODULUS)
        self.d_model = d_model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def step_logits(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_repr = self.encoder.encode_state(state)
        command_repr = self.encoder.encode_command(command, a, b, dst)
        hidden = self.cell(torch.cat([state_repr, command_repr], dim=-1), hidden)
        return self.head(hidden), hidden

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        hidden = torch.zeros(
            batch.initial.shape[0], self.d_model, device=batch.initial.device
        )
        losses = []
        for t in range(batch.depth):
            state = batch.initial if t == 0 else batch.target_states[:, t - 1]
            dst = batch.dst[:, t]
            target = batch.target_states[:, t].gather(1, dst[:, None]).squeeze(1)
            logits, hidden = self.step_logits(
                state,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                dst,
                hidden,
            )
            losses.append(F.cross_entropy(logits, target))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def rollout(self, batch: ProgramBatch) -> torch.Tensor:
        hidden = torch.zeros(
            batch.initial.shape[0], self.d_model, device=batch.initial.device
        )
        state = batch.initial.clone()
        states = []
        for t in range(batch.depth):
            dst = batch.dst[:, t]
            logits, hidden = self.step_logits(
                state,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                dst,
                hidden,
            )
            value = logits.argmax(dim=-1)
            state = state.scatter(1, dst[:, None], value[:, None])
            states.append(state.clone())
        return torch.stack(states, dim=1)
