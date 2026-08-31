from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_REGISTERS, VALUE_MODULUS, ProgramBatch, StateCommandEncoder


class StateAccessGRUControl(nn.Module):
    """Recurrent control with current-state access but no direct indexed-value features."""

    def __init__(self, d_model: int = 112):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.cell = nn.GRUCell(2 * d_model, d_model)
        self.head = nn.Linear(d_model, VALUE_MODULUS)
        self.d_model = d_model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def step_logits(self, state, command, a, b, dst, hidden):
        state_repr = self.encoder.encode_state(state)
        command_repr = self.encoder.encode_command(command, a, b, dst)
        hidden = self.cell(torch.cat([state_repr, command_repr], dim=-1), hidden)
        return self.head(hidden), hidden

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        hidden = torch.zeros(batch.initial.shape[0], self.d_model, device=batch.initial.device)
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
        hidden = torch.zeros(batch.initial.shape[0], self.d_model, device=batch.initial.device)
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


class FeatureMatchedStateGRUControl(nn.Module):
    """GRU control with exactly the transition features exposed to SharedTransitionModel.

    Both models receive the whole-state representation, command representation, direct embeddings
    for state[a], state[b], state[dst], and the destination-register embedding. The control also
    retains a latent recurrent hidden state, making it conservative with respect to memory capacity.
    """

    def __init__(self, d_model: int = 88):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.cell = nn.GRUCell(6 * d_model, d_model)
        self.head = nn.Linear(d_model, VALUE_MODULUS)
        self.d_model = d_model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _gather(state: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return state.gather(1, index[:, None]).squeeze(1)

    def step_logits(self, state, command, a, b, dst, hidden):
        state_repr = self.encoder.encode_state(state)
        command_repr = self.encoder.encode_command(command, a, b, dst)
        value_a = self.encoder.value(self._gather(state, a))
        value_b = self.encoder.value(self._gather(state, b))
        value_dst = self.encoder.value(self._gather(state, dst))
        register_dst = self.encoder.register(dst)
        features = torch.cat(
            [state_repr, command_repr, value_a, value_b, value_dst, register_dst], dim=-1
        )
        hidden = self.cell(features, hidden)
        return self.head(hidden), hidden

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        hidden = torch.zeros(batch.initial.shape[0], self.d_model, device=batch.initial.device)
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
        hidden = torch.zeros(batch.initial.shape[0], self.d_model, device=batch.initial.device)
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
