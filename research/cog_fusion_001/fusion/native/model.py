"""COG connected v2: a causal record/controller loop after the v1 backbone.

The six original local blocks remain independent of the record board. A new
small controller is the internal state connected to reads and future writes.
The primary repair retains VALUE/HOLD operations; it does not add native COPY.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__:
    from .scripted_core import recurrent_core
    from .base_model import (COGTinyLM as _V1LM, ModelConfig as _V1Config,
                             LatentRecordBoard, LocalAffineBlock,
                             affine_scan, detach_state, hard_categorical)
else:
    from scripted_core import recurrent_core
    from base_model import (COGTinyLM as _V1LM, ModelConfig as _V1Config,
                            LatentRecordBoard, LocalAffineBlock,
                            affine_scan, detach_state, hard_categorical)


ARCHITECTURE_ID = "cog-unified-connected-v2"
PARENT_ARCHITECTURE_ID = "cog-unified-tiny-latent-v1"

# This is an explicit migration allowlist, not a prefix-based permissive load.
NEW_BOARD_PARAMETER_NAMES = frozenset({
    "controller_input.weight",
    "controller.weight_ih", "controller.weight_hh",
    "controller.bias_ih", "controller.bias_hh",
    "query_entity_feedback.weight", "query_relation_feedback.weight",
    "write_entity_feedback.weight", "write_relation_feedback.weight",
    "write_commit_feedback.weight", "write_value_feedback.weight",
    "read_to_value.weight", "controller_output.weight",
})
NEW_PARAMETER_NAMES = frozenset(
    "record_board." + name for name in NEW_BOARD_PARAMETER_NAMES)


@dataclass(frozen=True)
class ConnectedConfig:
    vocab_size: int
    d_model: int = 122
    n_layers: int = 6
    ffn_mult: int = 2
    n_entities: int = 4
    n_relations: int = 4
    memory_dim: int = 64
    dropout: float = 0.1
    controller_dim: int = 64
    architecture_id: str = ARCHITECTURE_ID

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "ffn_mult",
                     "n_entities", "n_relations", "memory_dim", "controller_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.architecture_id != ARCHITECTURE_ID:
            raise ValueError("unsupported architecture_id")


class ConnectedRecordBoard(LatentRecordBoard):
    """Read-before-write board with a small recurrent controller.

    The input-only projections are batched across time. The controller/board
    loop is sequential because later queries and writes can depend on earlier
    retrieved records. No state is detached within this call.

    ``feedback_enabled=False`` zeros the read passed into the controller and
    writer while retaining the direct read-to-output residual. A comparison
    that removes *all historical* feedback must start with the same fresh
    controller in each arm. A previously connected controller can still carry
    information when the flag is switched midway through a stream.

    Oracle overrides are an explicit diagnostic facility, not part of native
    language-model training or evaluation. They substitute discrete routes or
    commits; no surrogate route gradient is retained at overridden locations.
    """

    _correction_modules = (
        "query_entity_feedback", "query_relation_feedback",
        "write_entity_feedback", "write_relation_feedback",
        "write_commit_feedback", "write_value_feedback", "read_to_value",
        "controller_output",
    )

    def __init__(self, cfg: ConnectedConfig, *, initialize: bool = True) -> None:
        super().__init__(cfg)
        self.use_scripted_core = True
        d, m, c = cfg.d_model, cfg.memory_dim, cfg.controller_dim
        self.controller_input = nn.Linear(d, c, bias=False)
        self.controller = nn.GRUCell(c + m, c)
        self.query_entity_feedback = nn.Linear(c, cfg.n_entities, bias=False)
        self.query_relation_feedback = nn.Linear(c, cfg.n_relations, bias=False)
        self.write_entity_feedback = nn.Linear(c, cfg.n_entities, bias=False)
        self.write_relation_feedback = nn.Linear(c, cfg.n_relations, bias=False)
        self.write_commit_feedback = nn.Linear(c, 1, bias=False)
        self.write_value_feedback = nn.Linear(c, m, bias=False)
        self.read_to_value = nn.Linear(m, m, bias=False)
        self.controller_output = nn.Linear(c, d, bias=False)
        if initialize:
            self.apply(_V1LM._initialize)
            nn.init.constant_(self.write_commit.bias, 1.0)
            nn.init.constant_(self.read_gate.bias, -1.0)
        self.reset_feedback_heads()

    def reset_feedback_heads(self) -> None:
        """Start with the v1 input/output function, while leaving c trainable."""
        for name in self._correction_modules:
            nn.init.zeros_(getattr(self, name).weight)

    def init_controller(self, batch_size: int,
                        device: torch.device | str | None = None) -> Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(batch_size, self.cfg.controller_dim,
                           device=device if device is not None else self.norm.weight.device,
                           dtype=torch.float32)

    def _validate_inputs(self, features: Tensor, memory: Tensor,
                         controller: Tensor) -> None:
        cfg = self.cfg
        if (features.ndim != 3 or min(features.shape[:2]) < 1
                or features.size(-1) != cfg.d_model
                or not features.is_floating_point()):
            raise ValueError("features must be nonempty floating [B,T,d_model]")
        expected_memory = (features.size(0), cfg.n_entities * cfg.n_relations,
                           cfg.memory_dim)
        if memory.shape != expected_memory or controller.shape != (
                features.size(0), cfg.controller_dim):
            raise ValueError("memory/controller shape mismatch")
        for value in (memory, controller):
            if value.device != features.device or not value.is_floating_point():
                raise ValueError("memory/controller device or dtype mismatch")

    def _prepare_overrides(self, overrides: Mapping[str, Tensor] | None,
                           features: Tensor) -> dict[str, Tensor]:
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise ValueError("overrides must be a mapping")
        if set(overrides) - {"query_slots", "write_slots", "commits"}:
            raise ValueError("unknown diagnostic override")
        prepared: dict[str, Tensor] = {}
        for name, value in overrides.items():
            if not isinstance(value, Tensor) or value.device != features.device:
                raise ValueError("override tensors must share the features device")
            if name == "commits" and value.shape == (*features.shape[:2], 1):
                value = value.squeeze(-1)
            if value.shape != features.shape[:2]:
                raise ValueError("overrides must have shape [B,T]")
            if name == "commits":
                if not bool(((value == -1) | (value == 0) | (value == 1)).all()):
                    raise ValueError("commit overrides must be -1, 0, or 1")
                prepared[name] = value.float()
            else:
                if value.dtype not in (torch.int32, torch.int64):
                    raise ValueError("address overrides must contain integer slot indices")
                slots = self.cfg.n_entities * self.cfg.n_relations
                if not bool(((value >= -1) & (value < slots)).all()):
                    raise ValueError("slot override out of range")
                prepared[name] = value.long()
        return prepared

    @staticmethod
    def _override_address(native: Tensor, override: Tensor | None) -> Tensor:
        if override is None:
            return native
        forced = F.one_hot(override.clamp_min(0), native.size(-1)).to(native.dtype)
        return torch.where((override >= 0).unsqueeze(-1), forced, native)

    def forward(self, features: Tensor, memory: Tensor,
                controller: Tensor | None = None, *,
                feedback_enabled: bool = True, memory_enabled: bool = True,
                return_trace: bool = False,
                overrides: Mapping[str, Tensor] | None = None,
                ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor] | None]:
        if controller is None:
            controller = self.init_controller(features.size(0), features.device)
        self._validate_inputs(features, memory, controller)
        oracle = self._prepare_overrides(overrides, features)
        cfg = self.cfg
        u = self.norm(features)
        # The expensive input-only projections retain batched [B,T,...] work.
        qe_base = self.query_entity(u).float()
        qr_base = self.query_relation(u).float()
        we_base = self.write_entity(u).float()
        wr_base = self.write_relation(u).float()
        wc_base = self.write_commit(u).float()
        wv_base = self.write_value(u).float()
        controller_inputs = self.controller_input(u)
        read_gates = self.read_gate(u).sigmoid()

        # Concatenating small projection matrices once reduces per-step dispatch
        # overhead without changing their parameter identities or gradients.
        query_feedback_weight = torch.cat((self.query_entity_feedback.weight,
                                           self.query_relation_feedback.weight), 0)
        writer_feedback_weight = torch.cat((self.write_entity_feedback.weight,
                                            self.write_relation_feedback.weight,
                                            self.write_commit_feedback.weight,
                                            self.write_value_feedback.weight), 0)
        if (self.use_scripted_core and not return_trace and overrides is None
                and features.dtype == torch.float32 and features.device.type == "cpu"
                and self.controller.weight_ih.dtype == torch.float32):
            all_reads, all_controllers, next_memory, next_controller = recurrent_core(
                qe_base, qr_base, we_base, wr_base, wc_base, wv_base,
                controller_inputs, memory, controller,
                query_feedback_weight, writer_feedback_weight,
                self.controller.weight_ih, self.controller.weight_hh,
                self.controller.bias_ih, self.controller.bias_hh,
                self.read_to_value.weight, cfg.n_entities, cfg.n_relations,
                cfg.memory_dim, feedback_enabled, memory_enabled)
            output = (features + self.read_value(all_reads) * read_gates
                      + self.controller_output(all_controllers))
            return output, next_memory, next_controller, None
        current_memory, c = memory.float(), controller.float()
        reads: list[Tensor] = []
        controllers: list[Tensor] = []
        history: dict[str, list[Tensor]] | None = None
        if return_trace:
            history = {key: [] for key in (
                "internal_reads", "query_entity_logits", "query_relation_logits",
                "write_entity_logits", "write_relation_logits", "write_commit_logits",
                "write_value_logits", "query_addresses", "write_addresses",
                "query_slots", "write_slots", "commit_probabilities", "commits",
                "proposed_values",
            )}

        for t in range(features.size(1)):
            q_feedback = F.linear(c.to(query_feedback_weight.dtype),
                                  query_feedback_weight).float()
            qe_delta, qr_delta = q_feedback.split((cfg.n_entities, cfg.n_relations), -1)
            qe, qr = qe_base[:, t] + qe_delta, qr_base[:, t] + qr_delta
            query = self._address(qe, qr)
            query = self._override_address(
                query, oracle["query_slots"][:, t] if "query_slots" in oracle else None)
            if memory_enabled:
                # Reads are from M[t-1], before the VALUE/HOLD update at t.
                read = (query.unsqueeze(-1) * current_memory).sum(-2)
            else:
                read = current_memory.new_zeros((features.size(0), cfg.memory_dim))
            internal_read = read if feedback_enabled and memory_enabled else torch.zeros_like(read)
            controller_input = torch.cat((controller_inputs[:, t].float(), internal_read), -1)
            c = self.controller(controller_input.to(self.controller.weight_ih.dtype),
                                c.to(self.controller.weight_hh.dtype)).float()

            w_feedback = F.linear(c.to(writer_feedback_weight.dtype),
                                  writer_feedback_weight).float()
            we_delta, wr_delta, wc_delta, wv_delta = w_feedback.split(
                (cfg.n_entities, cfg.n_relations, 1, cfg.memory_dim), -1)
            we, wr = we_base[:, t] + we_delta, wr_base[:, t] + wr_delta
            wc = wc_base[:, t] + wc_delta
            wv = (wv_base[:, t] + wv_delta
                  + self.read_to_value(internal_read.to(self.read_to_value.weight.dtype)).float())
            address = self._address(we, wr)
            address = self._override_address(
                address, oracle["write_slots"][:, t] if "write_slots" in oracle else None)
            probability = wc.sigmoid()
            hard_commit = (probability >= 0.5).to(probability.dtype)
            commit = hard_commit + (probability - probability.detach())
            if "commits" in oracle:
                forced_commit = oracle["commits"][:, t].unsqueeze(-1)
                commit = torch.where(forced_commit >= 0, forced_commit, commit)
            value = wv.tanh()
            if memory_enabled:
                weight = (commit * address).unsqueeze(-1)
                current_memory = (1.0 - weight) * current_memory + weight * value.unsqueeze(-2)

            reads.append(read)
            controllers.append(c)
            if history is not None:
                values = {
                    "internal_reads": internal_read,
                    "query_entity_logits": qe, "query_relation_logits": qr,
                    "write_entity_logits": we, "write_relation_logits": wr,
                    "write_commit_logits": wc, "write_value_logits": wv,
                    "query_addresses": query, "write_addresses": address,
                    "query_slots": query.argmax(-1), "write_slots": address.argmax(-1),
                    "commit_probabilities": probability, "commits": commit,
                    "proposed_values": value,
                }
                for key, value_tensor in values.items():
                    history[key].append(value_tensor)

        all_reads = torch.stack(reads, dim=1)
        all_controllers = torch.stack(controllers, dim=1)
        output = (features + self.read_value(all_reads.to(features.dtype)) * read_gates
                  + self.controller_output(all_controllers.to(features.dtype)))
        trace = None
        if history is not None:
            trace = {key: torch.stack(value_list, 1) for key, value_list in history.items()}
            trace["reads"] = all_reads
            trace["controllers"] = all_controllers
        # Clones prevent returned inference state from retaining a time-slice
        # backing allocation. They deliberately preserve autograd for training.
        return output, current_memory.clone(), c.clone(), trace


class ConnectedCOGTinyLM(_V1LM):
    """Checkpoint-compatible backbone plus the explicitly connected board."""

    def __init__(self, config: ConnectedConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.default_feedback_enabled = True
        self.migration_metadata: dict[str, Any] | None = None
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([LocalAffineBlock(config) for _ in range(config.n_layers)])
        self.record_board = ConnectedRecordBoard(config, initialize=False)
        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        scale = 1 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.out_proj.weight.mul_(scale)
                block.ffn_down.weight.mul_(scale)
                block.in_proj.bias[:config.d_model].copy_(
                    torch.linspace(-2.0, 3.0, config.d_model))
        nn.init.constant_(self.record_board.write_commit.bias, 1.0)
        nn.init.constant_(self.record_board.read_gate.bias, -1.0)
        self.record_board.reset_feedback_heads()

    def new_parameter_names(self) -> frozenset[str]:
        """Exact migrated-parameter complement, for auditable LR grouping."""
        available = dict(self.named_parameters())
        if not NEW_PARAMETER_NAMES <= available.keys():
            raise RuntimeError("new-parameter allowlist does not match architecture")
        return NEW_PARAMETER_NAMES

    def init_state(self, batch_size: int, device: torch.device | str | None = None,
                   dtype: torch.dtype | None = None) -> dict[str, Any]:
        state = super().init_state(batch_size, device, dtype)
        state["controller"] = self.record_board.init_controller(
            batch_size, device if device is not None else self.embedding.weight.device)
        return state

    def _validate_state(self, state: dict[str, Any], tokens: Tensor) -> None:
        if not isinstance(state, dict) or set(state) != {"layers", "memory", "controller"}:
            raise ValueError("state does not match connected-v2 geometry")
        super()._validate_state({"layers": state["layers"], "memory": state["memory"]}, tokens)
        controller = state["controller"]
        if (not isinstance(controller, Tensor)
                or controller.shape != (tokens.size(0), self.config.controller_dim)
                or controller.device != tokens.device or not controller.is_floating_point()):
            raise ValueError("controller state shape/device/dtype mismatch")

    def backbone_features(self, input_ids: Tensor, state: dict[str, Any] | None = None
                          ) -> tuple[Tensor, tuple[dict[str, Tensor], ...]]:
        """Run only the original local backbone; useful for frozen diagnostics."""
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
        return x, tuple(next_layers)

    def forward(self, input_ids: Tensor, state: dict[str, Any] | None = None, *,
                feedback_enabled: bool | None = None, memory_enabled: bool = True,
                return_trace: bool = False,
                overrides: Mapping[str, Tensor] | None = None,
                ) -> tuple[Tensor, dict[str, Any]] | tuple[Tensor, dict[str, Any], dict[str, Tensor]]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be nonempty [B,T] torch.long")
        if state is None:
            state = self.init_state(input_ids.size(0), input_ids.device)
        if feedback_enabled is None:
            feedback_enabled = self.default_feedback_enabled
        x, next_layers = self.backbone_features(input_ids, state)
        x, memory, controller, trace = self.record_board(
            x, state["memory"], state["controller"], feedback_enabled=feedback_enabled,
            memory_enabled=memory_enabled, return_trace=return_trace, overrides=overrides)
        logits = self.lm_head(self.final_norm(x)).float()
        next_state = {"layers": next_layers, "memory": memory, "controller": controller}
        if return_trace:
            assert trace is not None
            return logits, next_state, trace
        return logits, next_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def migrate_v1(checkpoint: str | PathLike[str] | Mapping[str, Any], *,
               seed: int | None = None, expected_parent_sha256: str | None = None,
               map_location: torch.device | str = "cpu", controller_dim: int = 64,
               ) -> tuple[ConnectedCOGTinyLM, dict[str, Any]]:
    """Strictly migrate an original v1 checkpoint, with explicit provenance.

    All old parameter names and shapes must match exactly. Only the 13 named
    new tensors may be missing. The new adapter/output heads are zero, so the
    migrated model initially has the original v1 logits up to roundoff.

    A filesystem checkpoint is hashed *before* loading. An in-memory payload
    cannot attest its original file hash; expected_parent_sha256 is therefore
    accepted only for a filesystem path. Load only trusted checkpoint files.
    """
    parent_path: str | None = None
    parent_hash: str | None = None
    if isinstance(checkpoint, (str, PathLike)):
        path = Path(checkpoint).expanduser().resolve()
        parent_path, parent_hash = str(path), _sha256(path)
        if expected_parent_sha256 is not None and parent_hash != expected_parent_sha256.lower():
            raise ValueError("parent checkpoint SHA256 mismatch")
        payload = torch.load(path, map_location=map_location, weights_only=True)
    elif isinstance(checkpoint, Mapping):
        if expected_parent_sha256 is not None:
            raise ValueError("cannot verify a file SHA256 from an in-memory checkpoint")
        payload = checkpoint
    else:
        raise TypeError("checkpoint must be a path or a checkpoint mapping")
    if not isinstance(payload, Mapping) or "model_config" not in payload or "model" not in payload:
        raise ValueError("expected a v1 checkpoint with model_config and model")
    parent_config = _V1Config(**dict(payload["model_config"]))
    config_values = asdict(parent_config)
    config_values.update(architecture_id=ARCHITECTURE_ID, controller_dim=controller_dim)
    if seed is not None:
        torch.manual_seed(seed)
    model = ConnectedCOGTinyLM(ConnectedConfig(**config_values))
    old_weights = payload["model"]
    if not isinstance(old_weights, Mapping):
        raise ValueError("checkpoint model must be a state-dict mapping")
    expected_old_keys = set(model.state_dict()) - NEW_PARAMETER_NAMES
    if set(old_weights) != expected_old_keys:
        missing = sorted(expected_old_keys - set(old_weights))
        unexpected = sorted(set(old_weights) - expected_old_keys)
        raise ValueError(f"v1 key mismatch: missing={missing}, unexpected={unexpected}")
    incompatible = model.load_state_dict(old_weights, strict=False)
    if set(incompatible.missing_keys) != NEW_PARAMETER_NAMES or incompatible.unexpected_keys:
        raise RuntimeError("migration exceeded its explicit new-key allowlist")
    new_names = sorted(model.new_parameter_names())
    parameters = dict(model.named_parameters())
    metadata = {
        "migration_format_version": 1,
        "architecture_id": ARCHITECTURE_ID,
        "parent_architecture_id": PARENT_ARCHITECTURE_ID,
        "parent_checkpoint_path": parent_path,
        "parent_checkpoint_sha256": parent_hash,
        "parent_step": payload.get("step"),
        "parent_training_targets_seen": payload.get("training_targets_seen"),
        "parent_source_sha256": payload.get("source_sha256"),
        "parent_dataset_sha256": payload.get("dataset_sha256"),
        "parent_tokenizer_sha256": payload.get("tokenizer_sha256"),
        "controller_dim": controller_dim,
        "new_parameter_names": new_names,
        "new_parameter_count": sum(parameters[name].numel() for name in new_names),
        "parameter_count": model.parameter_count(),
        "zero_initialized_corrections": list(model.record_board._correction_modules),
    }
    model.migration_metadata = metadata
    return model, metadata


def load_v1_checkpoint(path: str | PathLike[str], *,
                       expected_parent_sha256: str | None = None,
                       map_location: torch.device | str = "cpu",
                       ) -> tuple[ConnectedCOGTinyLM, dict[str, Any]]:
    return migrate_v1(path, expected_parent_sha256=expected_parent_sha256,
                      map_location=map_location)


# Explicit aliases for simple local experiment scripts. Their architecture ID
# and state geometry remain v2; these aliases do not make v1 checkpoints load
# silently without the migration helper.
ModelConfig = ConnectedConfig
ConnectedLM = ConnectedCOGTinyLM
COGConnectedLM = ConnectedCOGTinyLM

