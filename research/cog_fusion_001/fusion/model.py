"""Trainable selective-memory COG with a genuine character-output connection.

The native local affine recurrence and 65-character head are reused exactly.
The event interpreter consumes the same continuously computed features, while
its GRU resets only at explicit event boundaries. Discrete factual writes and
reads are executed outside this module. No labels or answer strings enter the
decoder: its evidence is at most two categorical record values.

The optional hybrid scores only the caller's supplied candidates. Its bucket
classifier is over a fixed 16-address codebook; that comparison work must be
reported separately from the candidate budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .native.base_model import affine_scan
from .native.model import ConnectedCOGTinyLM, ConnectedConfig

SCHEMA = "cog-fusion-001-v1"
NATIVE_SHA256 = "d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c"
NATIVE_SOURCES = {
    "model.py": "f15e7d5430bd1df83a1264ebbec63576b818733386d6fb0f1d8c1e2a930370f9",
    "base_model.py": "f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa",
    "scripted_core.py": "6c00b2c4425a5309927fb3aba4e53add670e79c88a7b051fbe963b6b060f99e3",
}


@dataclass(frozen=True)
class FusionConfig:
    semantic_dim: int = 64
    controller_dim: int = 32
    evidence_dim: int = 32
    key_dim: int = 32
    n_entities: int = 16
    n_relations: int = 2
    n_buckets: int = 16
    packet_size: int = 2
    native_board_enabled: bool = False

    def __post_init__(self):
        for name in ("semantic_dim", "controller_dim", "evidence_dim", "key_dim",
                     "n_entities", "n_relations", "n_buckets", "packet_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.n_entities, self.n_relations, self.n_buckets, self.packet_size) != (16, 2, 16, 2):
            raise ValueError("this protocol fixes 16 entities, two relations, 16 buckets and two values")
        if type(self.native_board_enabled) is not bool:
            raise ValueError("native_board_enabled must be boolean")


def state_map(state: Any, function):
    if isinstance(state, Tensor):
        return function(state)
    if isinstance(state, dict):
        return {key: state_map(value, function) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(state_map(value, function) for value in state)
    if isinstance(state, list):
        return [state_map(value, function) for value in state]
    raise TypeError("continuation state must consist only of tensors and containers")


def detach_state(state: dict) -> dict:
    return state_map(state, lambda x: x.detach())


def clone_state(state: dict) -> dict:
    return state_map(state, lambda x: x.clone())


def state_bytes(state: dict) -> int:
    total = 0
    def count(value):
        nonlocal total
        total += value.numel() * value.element_size()
        return value
    state_map(state, count)
    return total


class FusionModel(nn.Module):
    """COG backbone, learned event semantics, sparse route, and neural decoder."""

    def __init__(self, native: ConnectedCOGTinyLM, chars: Sequence[str], *,
                 config: FusionConfig | None = None, memory_enabled: bool = True,
                 admission: Mapping[str, Any] | None = None):
        super().__init__()
        self.native = native
        self.config = config or FusionConfig()
        cfg = self.config
        self.chars = tuple(chars)
        if len(set(self.chars)) != len(self.chars) or len(self.chars) != native.config.vocab_size:
            raise ValueError("tokenizer does not match the native output head")
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.memory_enabled = bool(memory_enabled)
        self.admission = dict(admission or {})
        d, s, c, e, k = (native.config.d_model, cfg.semantic_dim,
                         cfg.controller_dim, cfg.evidence_dim, cfg.key_dim)
        # This small accumulator resets per declared event, not the backbone.
        self.event_norm = nn.RMSNorm(d, eps=1e-5)
        self.event_encoder = nn.GRU(d, s, batch_first=True)
        self.semantic_heads = nn.ModuleDict({
            "operation": nn.Linear(s, 3),
            "entity": nn.Linear(s, cfg.n_entities),
            "relation": nn.Linear(s, cfg.n_relations),
            "value": nn.Linear(s, cfg.n_entities + 1),
            "query_plan": nn.Linear(s, 3),
            "query_entity": nn.Linear(s, cfg.n_entities),
        })
        self.value_embedding = nn.Embedding(cfg.n_entities + 1, e)
        self.event_to_decode = nn.Linear(s, c, bias=False)
        self.decoder_controller = nn.GRU(d + cfg.packet_size * e + c, c, batch_first=True)
        self.evidence_output = nn.Linear(cfg.packet_size * e, d, bias=False)
        self.controller_output = nn.Linear(c, d, bias=False)
        self.previous_value_embedding = nn.Embedding(cfg.n_entities + 1, k)
        self.hop_embedding = nn.Embedding(2, k)
        self.route_query = nn.Linear(s + 2 * k, k)
        self.bucket_head = nn.Linear(k, cfg.n_buckets)
        self.key_entity = nn.Embedding(cfg.n_entities, k)
        self.key_relation = nn.Embedding(cfg.n_relations, k)
        self.route_scale = nn.Parameter(torch.tensor(1.0))
        # Matched arms initially share the unmodified local language logits.
        nn.init.zeros_(self.evidence_output.weight)
        nn.init.zeros_(self.controller_output.weight)

    @classmethod
    def from_native(cls, path: str | Path, seed: int = 17, *, memory_enabled: bool = True,
                    config: FusionConfig | None = None) -> "FusionModel":
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != NATIVE_SHA256:
            raise ValueError("native checkpoint is not the prospectively selected checkpoint")
        observed = {name: hashlib.sha256((Path(__file__).parent / "native" / name).read_bytes()).hexdigest()
                    for name in NATIVE_SOURCES}
        if observed != NATIVE_SOURCES:
            raise ValueError("vendored native source identity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for name, value in observed.items():
            matching = [sha for source, sha in payload["source_sha256"].items() if Path(source).name == name]
            if matching != [value]:
                raise ValueError("native checkpoint does not bind the loaded source")
        # Seed affects only new trainable interfaces; pretrained weights are exact.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            native = ConnectedCOGTinyLM(ConnectedConfig(**payload["model_config"]))
            native.load_state_dict(payload["model"], strict=True)
            if native.parameter_count() != 993574:
                raise ValueError("native parameter budget mismatch")
            return cls(native, payload["tokenizer"]["chars"], config=config,
                       memory_enabled=memory_enabled,
                       admission={"native_checkpoint_sha256": digest,
                                  "native_source_sha256": observed,
                                  "native_seed": payload["seed"], "native_step": payload["step"],
                                  "new_interface_seed": seed})

    def checkpoint_payload(self, **metadata) -> dict:
        return {"schema": SCHEMA, "native_config": asdict(self.native.config),
                "config": asdict(self.config), "chars": list(self.chars),
                "memory_enabled": self.memory_enabled, "admission": self.admission,
                "model": self.state_dict(), "metadata": metadata}

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "FusionModel":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != SCHEMA:
            raise ValueError("incompatible fusion checkpoint schema")
        native = ConnectedCOGTinyLM(ConnectedConfig(**payload["native_config"]))
        model = cls(native, payload["chars"], config=FusionConfig(**payload["config"]),
                    memory_enabled=payload["memory_enabled"], admission=payload["admission"])
        model.load_state_dict(payload["model"], strict=True)
        return model

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        native = self.native.parameter_count()
        inactive = (0 if self.config.native_board_enabled else
                    sum(parameter.numel() for parameter in self.native.record_board.parameters()))
        return {"total": total, "native": native, "new_interfaces": total - native,
                "inactive_native_board": inactive, "active": total - inactive}

    def model_identity(self) -> str:
        digest = hashlib.sha256(json.dumps({"schema": SCHEMA, "config": asdict(self.config),
                                           "native": asdict(self.native.config), "chars": self.chars,
                                           "memory_enabled": self.memory_enabled}, sort_keys=True).encode())
        for name, tensor in self.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def init_state(self, batch: int, device=None) -> dict:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        device = device or self.native.embedding.weight.device
        return {"native": self.native.init_state(batch, device),
                "decoder": torch.zeros(batch, self.config.controller_dim, device=device),
                "event": torch.zeros(batch, self.config.semantic_dim, device=device)}

    def validate_state(self, state: dict, tokens: Tensor, *, finite: bool = False):
        if not isinstance(state, dict) or set(state) != {"native", "decoder", "event"}:
            raise ValueError("incompatible continuation state")
        self.native._validate_state(state["native"], tokens)
        for key, width in (("decoder", self.config.controller_dim), ("event", self.config.semantic_dim)):
            value = state[key]
            if (not isinstance(value, Tensor) or value.shape != (tokens.size(0), width)
                    or value.device != tokens.device or value.dtype != self.native.embedding.weight.dtype):
                raise ValueError(f"invalid {key} state")
        if finite:
            def check(value):
                if not torch.isfinite(value).all():
                    raise ValueError("nonfinite continuation state")
                return value
            state_map(state, check)

    def encode_texts(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        if not texts or any(not text for text in texts):
            raise ValueError("texts must be nonempty")
        unknown = set().union(*(set(text) for text in texts)) - set(self.stoi)
        if unknown:
            raise ValueError(f"characters outside native vocabulary: {sorted(unknown)!r}")
        device = self.native.embedding.weight.device
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long, device=device)
        tokens = torch.full((len(texts), int(lengths.max())), self.stoi[" "], dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            tokens[row, :len(text)] = torch.tensor([self.stoi[char] for char in text], device=device)
        return tokens, lengths

    def _validate_tokens(self, tokens: Tensor, lengths: Tensor | None):
        if (not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long
                or min(tokens.shape) < 1 or tokens.device != self.native.embedding.weight.device):
            raise ValueError("tokens must be nonempty [B,T] int64 on model device")
        if lengths is None:
            lengths = torch.full((tokens.size(0),), tokens.size(1), dtype=torch.long, device=tokens.device)
        if (not isinstance(lengths, Tensor) or lengths.shape != (tokens.size(0),)
                or lengths.dtype != torch.long or lengths.device != tokens.device
                or not bool(((lengths >= 1) & (lengths <= tokens.size(1))).all())):
            raise ValueError("lengths must identify a nonempty valid prefix per row")
        return lengths

    @staticmethod
    def _last(values: Tensor, lengths: Tensor) -> Tensor:
        return values[torch.arange(values.size(0), device=values.device), lengths - 1].clone()

    def _backbone(self, tokens: Tensor, state: dict | None, lengths: Tensor | None = None):
        lengths = self._validate_tokens(tokens, lengths)
        state = self.init_state(tokens.size(0), tokens.device) if state is None else state
        self.validate_state(state, tokens)
        x = self.native.embedding(tokens)
        layers = []
        # Same operations as native LocalAffineBlock, with true-length gathering.
        # Padded suffix outputs never affect earlier causal outputs or carried state.
        for block, previous_state in zip(self.native.blocks, state["native"]["layers"]):
            u = block.norm(x)
            previous = torch.cat((previous_state["previous"].unsqueeze(1), u[:, :-1]), dim=1)
            mixed = u + torch.tanh(block.previous_gain) * previous
            decay_logits, value, output_gate = block.in_proj(mixed).chunk(3, dim=-1)
            decay = 0.90 + 0.0998 * torch.sigmoid(decay_logits.float())
            innovation = (1.0 - decay) * torch.tanh(value.float())
            hidden = affine_scan(decay, innovation, previous_state["hidden"])
            delta = block.out_proj((hidden * torch.sigmoid(output_gate.float())).to(x.dtype))
            x = x + block.dropout(delta)
            gate, value_ff = block.ffn_up(block.ffn_norm(x)).chunk(2, dim=-1)
            x = x + block.dropout(block.ffn_down(F.silu(gate) * value_ff))
            layers.append({"hidden": self._last(hidden, lengths), "previous": self._last(u, lengths)})
        native_state = {**state["native"], "layers": tuple(layers)}
        if self.config.native_board_enabled:
            # Native board's public API returns only the last padded state;
            # per-row slicing is required in this explicitly optional slow arm.
            rows, memories, controllers = [], [], []
            for row, length in enumerate(lengths.tolist()):
                part, memory, controller, _ = self.native.record_board(
                    x[row:row + 1, :length], state["native"]["memory"][row:row + 1],
                    state["native"]["controller"][row:row + 1])
                rows.append(F.pad(part, (0, 0, 0, x.size(1) - length)))
                memories.append(memory); controllers.append(controller)
            x = torch.cat(rows)
            native_state.update(memory=torch.cat(memories), controller=torch.cat(controllers))
        return x, {**state, "native": native_state}, lengths

    def encode_event(self, tokens: Tensor, lengths: Tensor | None = None, state: dict | None = None):
        x, next_state, lengths = self._backbone(tokens, state, lengths)
        # A zero initial event accumulator is intentional and deployment matched.
        features, _ = self.event_encoder(self.event_norm(x))
        feature = self._last(features, lengths)
        semantics = {name: head(feature) for name, head in self.semantic_heads.items()}
        return feature, semantics, {**next_state, "event": feature}

    def _evidence(self, evidence_values: Tensor, evidence_mask: Tensor, batch: int,
                  memory_enabled: bool | None) -> Tensor:
        shape = (batch, self.config.packet_size)
        if (evidence_values.shape != shape or evidence_values.dtype != torch.long
                or evidence_mask.shape != shape or evidence_mask.dtype != torch.bool
                or evidence_values.device != self.native.embedding.weight.device
                or evidence_mask.device != evidence_values.device):
            raise ValueError("evidence must be int64 values and boolean mask with shape [B,2]")
        if not bool(((evidence_values >= 0) & (evidence_values <= self.config.n_entities)).all()):
            raise ValueError("evidence value outside finite vocabulary")
        enabled = self.memory_enabled if memory_enabled is None else bool(memory_enabled)
        if not enabled:
            return self.value_embedding.weight.new_zeros((batch, self.config.packet_size * self.config.evidence_dim))
        embedded = self.value_embedding(evidence_values)
        return (embedded * evidence_mask.unsqueeze(-1)).flatten(1)

    def decode(self, tokens: Tensor, state: dict | None, evidence_values: Tensor, evidence_mask: Tensor,
               *, memory_enabled: bool | None = None, lengths: Tensor | None = None):
        x, next_state, lengths = self._backbone(tokens, state, lengths)
        evidence = self._evidence(evidence_values, evidence_mask, tokens.size(0), memory_enabled)
        event = self.event_to_decode(next_state["event"])
        controller_input = torch.cat((x, evidence[:, None].expand(-1, x.size(1), -1),
                                      event[:, None].expand(-1, x.size(1), -1)), dim=-1)
        controls, _ = self.decoder_controller(controller_input, next_state["decoder"].unsqueeze(0))
        adjusted = x + self.evidence_output(evidence)[:, None] + self.controller_output(controls)
        logits = self.native.lm_head(self.native.final_norm(adjusted)).float()
        return logits, {**next_state, "decoder": self._last(controls, lengths)}

    def forward_lm(self, tokens: Tensor, state: dict | None = None, *, lengths: Tensor | None = None):
        x, next_state, _ = self._backbone(tokens, state, lengths)
        return self.native.lm_head(self.native.final_norm(x)).float(), next_state

    def route_logits(self, query_features: Tensor, previous_value: Tensor | None = None, hop: int = 0):
        if query_features.ndim != 2 or query_features.size(1) != self.config.semantic_dim or hop not in (0, 1):
            raise ValueError("invalid route features or hop")
        batch = query_features.size(0)
        if previous_value is None:
            previous_value = torch.full((batch,), self.config.n_entities, dtype=torch.long,
                                        device=query_features.device)
        if previous_value.shape != (batch,) or previous_value.dtype != torch.long:
            raise ValueError("previous_value must contain the actual selected first-hop values")
        hop_ids = torch.full((batch,), hop, dtype=torch.long, device=query_features.device)
        query = torch.tanh(self.route_query(torch.cat((query_features,
                            self.previous_value_embedding(previous_value), self.hop_embedding(hop_ids)), dim=-1)))
        return self.bucket_head(query), query

    def candidate_scores(self, query: Tensor, candidate_entities: Tensor, candidate_relations: Tensor):
        if (candidate_entities.ndim != 2 or candidate_entities.shape != candidate_relations.shape
                or candidate_entities.size(0) != query.size(0)
                or candidate_entities.dtype != torch.long or candidate_relations.dtype != torch.long):
            raise ValueError("candidate identities must have matching [B,K] integer shapes")
        keys = self.key_entity(candidate_entities) + self.key_relation(candidate_relations)
        return (F.normalize(query, dim=-1)[:, None] * F.normalize(keys, dim=-1)).sum(-1) * self.route_scale.exp().clamp_max(100)

    @staticmethod
    def bucket(entity: Tensor | int, relation: Tensor | int):
        return (entity + 7 * relation) % 16
