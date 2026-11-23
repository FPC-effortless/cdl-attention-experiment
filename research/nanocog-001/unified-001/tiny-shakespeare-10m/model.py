"""COG-UNIFIED Tiny: native causal LM with hard-routed latent records.

This is the text-only trainable instantiation of the unified proposal, not a
claim that Shakespeare teaches semantic entities, COPY/LINK, or transactions.
No token-to-token QK attention or external SSM implementation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 400
    n_layers: int = 6
    ffn_mult: int = 2
    n_entities: int = 4
    n_relations: int = 4
    memory_dim: int = 64
    dropout: float = 0.1
    architecture_id: str = "cog-unified-tiny-latent-v1"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "ffn_mult",
                     "n_entities", "n_relations", "memory_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.architecture_id != "cog-unified-tiny-latent-v1":
            raise ValueError("unsupported architecture_id")


def detach_state(state: Any) -> Any:
    """Detach a state at a declared TBPTT boundary without mutating it."""
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    if isinstance(state, list):
        return [detach_state(v) for v in state]
    return state


def affine_scan(a: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Inclusive prefix of h[t]=a[t]*h[t-1]+b[t], along axis 1.

    Exact-algebra associative scan, implemented by parallel doubling. Work is
    O(T log T), not a claimed work-optimal accelerator kernel. Float32 scan
    accumulation avoids half-precision long-product underflow. Reassociation
    introduces ordinary floating-point roundoff, so parity is tolerance based.
    a may broadcast over b's final value axis (used by the record board).
    """
    if a.shape[:2] != b.shape[:2] or b.shape[1] < 1:
        raise ValueError("affine_scan expects matching nonempty [B,T,...]")
    aa, bb = a.float(), b.float()
    offset = 1
    while offset < bb.size(1):
        # Both right-hand sides use the same previous scan level.
        next_b = torch.cat((bb[:, :offset],
                            bb[:, offset:] + aa[:, offset:] * bb[:, :-offset]), 1)
        next_a = torch.cat((aa[:, :offset],
                            aa[:, offset:] * aa[:, :-offset]), 1)
        aa, bb = next_a, next_b
        offset *= 2
    return bb + aa * initial.float().unsqueeze(1)


def hard_categorical(logits: Tensor) -> Tensor:
    """Hard forward categorical route, softmax straight-through derivative.

    This is a biased gradient estimator, not differentiation of argmax.
    The softmax is over a fixed categorical address vocabulary, not token QK.
    """
    p = logits.float().softmax(-1)
    hard = F.one_hot(p.argmax(-1), p.size(-1)).to(p.dtype)
    return hard + (p - p.detach())


class LocalAffineBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, f = cfg.d_model, cfg.ffn_mult * cfg.d_model
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.previous_gain = nn.Parameter(torch.zeros(d))
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5)
        self.ffn_up = nn.Linear(d, 2 * f, bias=False)
        self.ffn_down = nn.Linear(f, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, state: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        u = self.norm(x)
        previous = torch.cat((state["previous"].unsqueeze(1), u[:, :-1]), dim=1)
        mixed = u + torch.tanh(self.previous_gain) * previous
        decay_logits, value, output_gate = self.in_proj(mixed).chunk(3, dim=-1)
        # A stable range is an implementation choice, not a semantic memory proof.
        decay = 0.90 + 0.0998 * torch.sigmoid(decay_logits.float())
        innovation = (1.0 - decay) * torch.tanh(value.float())
        hidden = affine_scan(decay, innovation, state["hidden"])
        delta = self.out_proj((hidden * torch.sigmoid(output_gate.float())).to(x.dtype))
        x = x + self.dropout(delta)
        gate, value_ff = self.ffn_up(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.dropout(self.ffn_down(F.silu(gate) * value_ff))
        # Clone avoids retaining the complete prefix merely through tensor views.
        return x, {"hidden": hidden[:, -1].clone(), "previous": u[:, -1].clone()}


class LatentRecordBoard(nn.Module):
    """Fixed entity-factor x relation-factor addresses, with VALUE/HOLD updates.

    Slots have persistent *latent categorical* identities. The coordinate names
    do not claim discovered human entities or relations. Queries read the board
    before the current write. All forward writes target exactly one slot or HOLD.
    Dense surrogate work during training is explicit and included in run cost.
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, m = cfg.d_model, cfg.memory_dim
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.query_entity = nn.Linear(d, cfg.n_entities)
        self.query_relation = nn.Linear(d, cfg.n_relations)
        self.write_entity = nn.Linear(d, cfg.n_entities)
        self.write_relation = nn.Linear(d, cfg.n_relations)
        self.write_commit = nn.Linear(d, 1)
        self.write_value = nn.Linear(d, m)
        self.read_value = nn.Linear(m, d, bias=False)
        self.read_gate = nn.Linear(d, 1)

    def _address(self, entity: Tensor, relation: Tensor) -> Tensor:
        e, r = hard_categorical(entity), hard_categorical(relation)
        return (e.unsqueeze(-1) * r.unsqueeze(-2)).flatten(-2)

    def forward(self, x: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        u = self.norm(x)
        query = self._address(self.query_entity(u), self.query_relation(u))
        address = self._address(self.write_entity(u), self.write_relation(u))
        probability = self.write_commit(u).float().sigmoid()
        hard = (probability >= 0.5).to(probability.dtype)
        commit = hard + (probability - probability.detach())
        weight = (commit * address).unsqueeze(-1)
        value = torch.tanh(self.write_value(u).float()).unsqueeze(-2)
        post_write = affine_scan(1.0 - weight, weight * value, memory)
        pre_write = torch.cat((memory.float().unsqueeze(1), post_write[:, :-1]), dim=1)
        read = (query.unsqueeze(-1) * pre_write).sum(-2)
        # Shared learned value interface trains against the actual written vectors.
        residual = self.read_value(read.to(x.dtype)) * self.read_gate(u).sigmoid()
        return x + residual, post_write[:, -1].clone()


class COGTinyLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([LocalAffineBlock(config) for _ in range(config.n_layers)])
        self.record_board = LatentRecordBoard(config)
        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # Residual projections remain nonzero so the first LM loss reaches routes.
        scale = 1 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.out_proj.weight.mul_(scale)
                block.ffn_down.weight.mul_(scale)
                # Diverse initial retention across coordinates.
                block.in_proj.bias[:config.d_model].copy_(
                    torch.linspace(-2.0, 3.0, config.d_model))
        nn.init.constant_(self.record_board.write_commit.bias, 1.0)
        nn.init.constant_(self.record_board.read_gate.bias, -1.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(self, batch_size: int, device: torch.device | str | None = None,
                   dtype: torch.dtype | None = None) -> dict[str, Any]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = device if device is not None else self.embedding.weight.device
        dtype = dtype if dtype is not None else self.embedding.weight.dtype
        d, cfg = self.config.d_model, self.config
        return {
            "layers": tuple({"hidden": torch.zeros(batch_size, d, device=device, dtype=torch.float32),
                             "previous": torch.zeros(batch_size, d, device=device, dtype=dtype)}
                            for _ in self.blocks),
            "memory": torch.zeros(batch_size, cfg.n_entities * cfg.n_relations, cfg.memory_dim,
                                  device=device, dtype=torch.float32),
        }

    def _validate_state(self, state: dict[str, Any], tokens: Tensor) -> None:
        b, d, cfg = tokens.size(0), self.config.d_model, self.config
        if set(state) != {"layers", "memory"} or len(state["layers"]) != cfg.n_layers:
            raise ValueError("state does not match model geometry")
        for layer in state["layers"]:
            if set(layer) != {"hidden", "previous"}:
                raise ValueError("malformed layer state")
            for name in ("hidden", "previous"):
                if layer[name].shape != (b, d) or layer[name].device != tokens.device:
                    raise ValueError("layer state batch/device mismatch")
                if not layer[name].is_floating_point():
                    raise ValueError("layer state must be floating point")
        if state["memory"].shape != (b, cfg.n_entities * cfg.n_relations, cfg.memory_dim):
            raise ValueError("memory shape mismatch")
        if state["memory"].device != tokens.device or not state["memory"].is_floating_point():
            raise ValueError("memory device/dtype mismatch")

    def forward(self, input_ids: Tensor, state: dict[str, Any] | None = None,
                *, memory_enabled: bool = True) -> tuple[Tensor, dict[str, Any]]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be nonempty [B,T] torch.long")
        if state is None:
            state = self.init_state(input_ids.size(0), input_ids.device)
        self._validate_state(state, input_ids)
        x = self.embedding(input_ids)
        next_layers = []
        for block, previous in zip(self.blocks, state["layers"]):
            x, next_state = block(x, previous)
            next_layers.append(next_state)
        if memory_enabled:
            x, memory = self.record_board(x, state["memory"])
        else:
            memory = state["memory"]
        logits = self.lm_head(self.final_norm(x)).float()
        return logits, {"layers": tuple(next_layers), "memory": memory}
