"""CSM-001: an experimental, learned, hard-addressed record rewrite backbone.

This is a broadly recurrent model, inspired by production systems and graph
rewriting. Neither architectural novelty nor sparse-kernel efficiency is claimed.
Training uses a biased straight-through categorical estimator. Forward selection
is hard in both training and evaluation; the implementation uses dense tensors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    width: int = 32
    slots: int = 8
    operators: int = 4
    inner: int = 64
    mode: str = "rewrite"  # rewrite | ring_write | no_links
    temperature: float = 1.0


@dataclass
class RecordState:
    values: Tensor
    links: Tensor
    cursor: Tensor
    history: Tensor
    position: int


def hard_choice(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Deterministic hard forward; softmax surrogate derivative in training."""
    probabilities = (logits / temperature).softmax(-1)
    chosen = F.one_hot(probabilities.argmax(-1), probabilities.shape[-1]).to(logits.dtype)
    return chosen + (probabilities - probabilities.detach())


class RecordRewrite(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.mode not in {"rewrite", "ring_write", "no_links"}:
            raise ValueError(config.mode)
        self.config = config
        v, d, s, o = config.vocab_size, config.width, config.slots, config.operators
        self.embedding = nn.Embedding(v, d)
        self.frontend = nn.Linear(3 * d, d)
        self.controller = nn.Linear(2 * d, d)
        self.first_address = nn.Linear(d, s)
        self.second_address = nn.Linear(2 * d, s)
        self.destination = nn.Linear(d, s)
        self.use_link = nn.Linear(d, 2)
        self.link_edit = nn.Linear(d, 2)
        self.operator = nn.Linear(d, o)
        self.operator_embedding = nn.Parameter(torch.randn(o, d) * 0.02)
        self.transform = nn.Sequential(
            nn.Linear(4 * d, config.inner), nn.Tanh(), nn.Linear(config.inner, d), nn.Tanh()
        )
        self.write_gate = nn.Linear(d, 1)
        self.value_edit = nn.Linear(d, 2)
        self.output = nn.Linear(d, v)
        nn.init.constant_(self.write_gate.bias, -0.5)
        with torch.no_grad():
            self.value_edit.bias.copy_(torch.tensor([0.0, 0.3]))

    def init_state(self, batch: int, device: torch.device | str) -> RecordState:
        d, s = self.config.width, self.config.slots
        dtype = self.embedding.weight.dtype
        values = torch.zeros(batch, s, d, device=device, dtype=dtype)
        links = torch.eye(s, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
        cursor = torch.zeros(batch, s, device=device, dtype=dtype)
        cursor[:, 0] = 1
        history = torch.zeros(batch, 2, d, device=device, dtype=dtype)
        return RecordState(values, links, cursor, history, 0)

    @staticmethod
    def _read(address: Tensor, values: Tensor) -> Tensor:
        return torch.bmm(address.unsqueeze(1), values).squeeze(1)

    def step(
        self, token_ids: Tensor, state: RecordState,
        interventions: dict[str, Any] | None = None,
        return_trace: bool = False,
    ):
        intervention = interventions or {}
        cfg = self.config
        raw = self.embedding(token_ids)
        front = torch.tanh(self.frontend(torch.cat((state.history.flatten(1), raw), -1)))
        active = self._read(state.cursor, state.values)
        control = torch.tanh(self.controller(torch.cat((front, active), -1)))
        choose = lambda logits: hard_choice(logits, cfg.temperature)
        first = choose(self.first_address(control))
        value_a = self._read(first, state.values)
        direct_second = choose(self.second_address(torch.cat((control, value_a), -1)))
        linked_second = self._read(first, state.links)
        link_choice = choose(self.use_link(control))
        if cfg.mode == "no_links" or intervention.get("links_off", False):
            second = direct_second
        else:
            second = link_choice[:, :1] * direct_second + link_choice[:, 1:] * linked_second
        value_b = self._read(second, state.values)
        operation = choose(self.operator(control))
        if intervention.get("constant_operator", False):
            operation = torch.zeros_like(operation)
            operation[:, 0] = 1
        operation_vector = operation @ self.operator_embedding
        proposal = self.transform(torch.cat((front, value_a, value_b, operation_vector), -1))
        destination = choose(self.destination(control))
        if cfg.mode == "ring_write":
            destination = torch.zeros_like(destination)
            destination[:, state.position % cfg.slots] = 1
        previous = self._read(destination, state.values)
        value_edit = choose(self.value_edit(control))[:, 1:]
        gate = value_edit * torch.sigmoid(self.write_gate(control))
        updated = previous + gate * (proposal - previous)
        values = state.values * (1 - destination.unsqueeze(-1)) + destination.unsqueeze(-1) * updated.unsqueeze(1)
        edit = choose(self.link_edit(control))[:, 1:]
        edge_mask = destination.unsqueeze(-1) * edit.unsqueeze(-1)
        links = state.links * (1 - edge_mask) + edge_mask * second.unsqueeze(1)
        next_state = RecordState(
            values, links, destination, torch.stack((state.history[:, 1], raw), 1), state.position + 1
        )
        logits = self.output(updated)
        if return_trace:
            return logits, next_state, {
                "first": first.argmax(-1), "second": second.argmax(-1),
                "destination": destination.argmax(-1), "operator": operation.argmax(-1),
                "uses_link": link_choice.argmax(-1), "edits_link": edit.squeeze(-1),
                "gate": gate.squeeze(-1), "edits_value": value_edit.squeeze(-1),
            }
        return logits, next_state

    def forward(self, input_ids: Tensor, interventions: dict[str, Any] | None = None) -> Tensor:
        state = self.init_state(input_ids.shape[0], input_ids.device)
        intervention = interventions or {}
        result = []
        reset_at = intervention.get("reset_at", None)
        corrupt_at = intervention.get("corrupt_at", None)
        for position in range(input_ids.shape[1]):
            if reset_at is not None and position == reset_at:
                state = self.init_state(input_ids.shape[0], input_ids.device)
                state.position = position
            if corrupt_at is not None and position == corrupt_at:
                if "zero_slot" in intervention:
                    values = state.values.clone()
                    values[:, int(intervention["zero_slot"])] = 0
                    state.values = values
                elif intervention.get("permute_values", False):
                    state.values = state.values.roll(1, 1)
            logits, state = self.step(input_ids[:, position], state, intervention)
            result.append(logits)
        return torch.stack(result, 1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self) -> int:
        count = self.parameter_count()
        if self.config.mode == "ring_write":
            count -= sum(p.numel() for p in self.destination.parameters())
        if self.config.mode == "no_links":
            # These heads cannot influence any output with links disabled.
            count -= sum(p.numel() for p in self.use_link.parameters())
            count -= sum(p.numel() for p in self.link_edit.parameters())
        return count

    def state_numel(self, sequence_length: int | None = None) -> int:
        d, s = self.config.width, self.config.slots
        return s * d + s * s + s + 2 * d

    def specification(self) -> dict[str, Any]:
        return {**asdict(self.config), "parameters": self.parameter_count(),
                "active_parameters": self.active_parameter_count(),
                "state_float32_bytes_per_example": self.state_numel() * 4,
                "state_position_integer": True,
                "dense_training_and_evaluation": True,
                "routing_estimator": "deterministic hard forward, softmax straight-through gradient"}
