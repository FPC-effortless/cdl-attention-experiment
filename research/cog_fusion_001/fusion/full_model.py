"""Optional read-compute-write extension; the locked v1 model is unchanged.

The persistent record controller receives actual selected numeric values. It
feeds proposed numeric writes, semantic write logits, subsequent retrieval keys
and the original character head. This establishes executable information paths;
it does not make their learned semantic use automatic.

The numeric composition diagnostic fits only RecordComputer. The frozen v1
checkpoint and the untrained feedback projections must be identified separately
from that fit. No jointly trained language/retrieval result is claimed for v2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .composition import RecordComputer
from .model import SCHEMA as BASE_SCHEMA, FusionConfig, FusionModel, state_bytes
from .native.model import ConnectedCOGTinyLM, ConnectedConfig


FULL_SCHEMA = "cog-fusion-optional-computation-v2"


@dataclass(frozen=True)
class FullConfig:
    value_dim: int = 8
    controller_dim: int = 32

    def __post_init__(self):
        if self.value_dim != 8 or self.controller_dim != 32:
            raise ValueError("this component protocol fixes eight values and 32 controller coordinates")


class FullFusionModel(nn.Module):
    """A composed model with explicit persistent state for selected-value work.

    The caller performs bounded record discovery and passes only the resulting
    payload to ``consume_selected``. No destination, support label or intended
    arithmetic answer enters that method. Routing requires the full state
    explicitly; silently forgetting the controller is rejected.
    """

    def __init__(self, base: FusionModel, *, computer: RecordComputer | None = None,
                 config: FullConfig | None = None, seed: int = 1701,
                 admission: Mapping[str, Any] | None = None):
        super().__init__()
        if not isinstance(base, FusionModel):
            raise TypeError("base must be the separately qualified FusionModel")
        self.base = base
        self.config = config or FullConfig()
        self.admission = dict(admission or {})
        # Local initialization must not consume a caller's sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.computer = computer if computer is not None else RecordComputer(
                value_dim=self.config.value_dim, hidden_dim=self.config.controller_dim)
            if (not isinstance(self.computer, RecordComputer)
                    or self.computer.value_dim != self.config.value_dim
                    or self.computer.hidden_dim != self.config.controller_dim):
                raise ValueError("attached computer dimensions differ from the declared full model")
            # Bias-free projections are nonzero: zero controller preserves v1,
            # while an actual read has immediately inspectable outgoing paths.
            self.semantic_feedback = nn.Linear(self.config.controller_dim,
                                               base.config.semantic_dim, bias=False)
            self.query_feedback = nn.Linear(self.config.controller_dim,
                                            base.config.key_dim, bias=False)
            self.character_feedback = nn.Linear(self.config.controller_dim,
                                                base.native.config.d_model, bias=False)
        self._parameter_mode = "joint_trainable"

    @classmethod
    def from_v1_checkpoint(cls, path: str | Path, *, seed: int = 1701,
                           computer: RecordComputer | None = None) -> "FullFusionModel":
        path = Path(path)
        base = FusionModel.from_checkpoint(path)
        return cls(base, computer=computer, seed=seed, admission={
            "base_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "base_model_identity": base.model_identity(),
            "component_seed": seed,
            "qualification": "optional component; not jointly trained language/retrieval",
        })

    @property
    def chars(self):
        return self.base.chars

    @property
    def stoi(self):
        return self.base.stoi

    @property
    def native(self):
        return self.base.native

    @property
    def memory_enabled(self):
        return self.base.memory_enabled

    def encode_texts(self, texts):
        return self.base.encode_texts(texts)

    def init_state(self, batch: int, device=None) -> dict:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        device = self.native.embedding.weight.device if device is None else device
        controller = self.computer.init_state(batch, device=device)
        return {"base": self.base.init_state(batch, device), "record_controller": controller}

    def _controller(self, state: dict, batch: int, *, finite: bool = True) -> Tensor:
        if not isinstance(state, dict) or set(state) != {"base", "record_controller"}:
            raise ValueError("full continuation requires base and record_controller")
        controller = state["record_controller"]
        parameter = self.native.embedding.weight
        if (not isinstance(controller, Tensor) or controller.shape != (batch, self.config.controller_dim)
                or controller.dtype != parameter.dtype or controller.device != parameter.device):
            raise ValueError("invalid record-controller geometry or execution type")
        if finite and not bool(torch.isfinite(controller).all()):
            raise ValueError("record controller must be finite")
        return controller

    def validate_state(self, state: dict, tokens: Tensor, *, finite: bool = False):
        self._controller(state, tokens.size(0), finite=finite)
        self.base.validate_state(state["base"], tokens, finite=finite)

    def state_tensor_bytes(self, batch: int = 1) -> int:
        return state_bytes(self.init_state(batch))

    def clear_controller(self, state: dict) -> dict:
        controller = self._controller(state, state["record_controller"].size(0))
        return {"base": state["base"], "record_controller": torch.zeros_like(controller)}

    def consume_selected(self, values: Tensor, state: dict) -> dict:
        """Consume actual selected payloads, retaining earlier reads in c."""
        if (not isinstance(values, Tensor) or values.ndim != 2
                or values.size(1) != self.config.value_dim):
            raise ValueError("selected numeric payload must have shape [B,8]")
        controller = self._controller(state, values.size(0))
        if (values.device != controller.device or values.dtype != controller.dtype
                or not bool(torch.isfinite(values).all())):
            raise ValueError("selected payload must be finite and match the controller type")
        updated = self.computer.consume(values, controller)
        result = {"base": state["base"], "record_controller": updated}
        self._controller(result, values.size(0))
        return result

    def propose_numeric_write(self, state: dict) -> Tensor:
        """Return a learned payload; the host still supplies/qualifies its destination."""
        controller = self._controller(state, state["record_controller"].size(0))
        value = self.computer.propose(controller)
        if value.shape != (controller.size(0), self.config.value_dim) or not bool(torch.isfinite(value).all()):
            raise ValueError("numeric writer returned an invalid proposed value")
        return value

    def semantic_write_logits(self, features: Tensor, *, state: dict) -> dict[str, Tensor]:
        """Actual semantic writer interface, conditioned directly on prior reads."""
        if features.ndim != 2 or features.size(1) != self.base.config.semantic_dim:
            raise ValueError("semantic feature shape differs from the base interpreter")
        controller = self._controller(state, features.size(0))
        feature = features + self.semantic_feedback(controller)
        return {name: self.base.semantic_heads[name](feature)
                for name in ("operation", "entity", "relation", "value")}

    def encode_event(self, tokens: Tensor, lengths: Tensor | None = None, state: dict | None = None):
        state = self.init_state(tokens.size(0), tokens.device) if state is None else state
        self.validate_state(state, tokens)
        features, _unused_base_semantics, next_base = self.base.encode_event(tokens, lengths, state["base"])
        adjusted = features + self.semantic_feedback(state["record_controller"])
        semantics = {name: head(adjusted) for name, head in self.base.semantic_heads.items()}
        next_base = {**next_base, "event": adjusted}
        return adjusted, semantics, {"base": next_base, "record_controller": state["record_controller"]}

    def route_logits(self, query_features: Tensor, previous_value: Tensor | None = None,
                     hop: int = 0, *, state: dict):
        controller = self._controller(state, query_features.size(0))
        _base_buckets, query = self.base.route_logits(query_features, previous_value=previous_value, hop=hop)
        query = query + self.query_feedback(controller)
        return self.base.bucket_head(query), query

    def candidate_scores(self, query: Tensor, candidate_entities: Tensor, candidate_relations: Tensor):
        return self.base.candidate_scores(query, candidate_entities, candidate_relations)

    @staticmethod
    def bucket(entity: Tensor | int, relation: Tensor | int):
        return FusionModel.bucket(entity, relation)

    def decode(self, tokens: Tensor, state: dict | None, evidence_values: Tensor, evidence_mask: Tensor,
               *, memory_enabled: bool | None = None, lengths: Tensor | None = None):
        state = self.init_state(tokens.size(0), tokens.device) if state is None else state
        self.validate_state(state, tokens)
        x, next_base, lengths = self.base._backbone(tokens, state["base"], lengths)
        evidence = self.base._evidence(evidence_values, evidence_mask, tokens.size(0), memory_enabled)
        event = self.base.event_to_decode(next_base["event"])
        inputs = torch.cat((x, evidence[:, None].expand(-1, x.size(1), -1),
                            event[:, None].expand(-1, x.size(1), -1)), dim=-1)
        controls, _ = self.base.decoder_controller(inputs, next_base["decoder"].unsqueeze(0))
        adjusted = (x + self.base.evidence_output(evidence)[:, None]
                    + self.base.controller_output(controls)
                    + self.character_feedback(state["record_controller"])[:, None])
        logits = self.native.lm_head(self.native.final_norm(adjusted)).float()
        next_base = {**next_base, "decoder": self.base._last(controls, lengths)}
        return logits, {"base": next_base, "record_controller": state["record_controller"]}

    def forward_lm(self, tokens: Tensor, state: dict | None = None, *, lengths: Tensor | None = None):
        state = self.init_state(tokens.size(0), tokens.device) if state is None else state
        self.validate_state(state, tokens)
        x, next_base, _ = self.base._backbone(tokens, state["base"], lengths)
        adjusted = x + self.character_feedback(state["record_controller"])[:, None]
        logits = self.native.lm_head(self.native.final_norm(adjusted)).float()
        return logits, {"base": next_base, "record_controller": state["record_controller"]}

    def numeric_training_only(self):
        """Only the attached numerical module is fitted by its separate diagnostic."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.computer.parameters():
            parameter.requires_grad_(True)
        self._parameter_mode = "numeric_component_only"
        self.base.eval()
        return self

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        base = self.base.parameter_summary()
        record = sum(parameter.numel() for parameter in self.computer.parameters())
        return {"total": total, "active": total - base["inactive_native_board"],
                "base_total": base["total"], "base_active": base["active"],
                "inactive_native_board": base["inactive_native_board"],
                "attached": total - base["total"], "record_computer": record,
                "feedback_projections": total - base["total"] - record,
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def model_identity(self) -> str:
        digest = hashlib.sha256(json.dumps({"schema": FULL_SCHEMA, "config": asdict(self.config),
                                           "base_identity": self.base.model_identity(),
                                           "computer_identity": self.computer.model_identity(),
                                           "full_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                          sort_keys=True).encode())
        for name, value in self.state_dict().items():
            if name.startswith("base."):
                continue
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def checkpoint_payload(self, **metadata) -> dict:
        # Store the base once. Explicit component dictionaries prevent a second
        # duplicate base state from disagreeing with the admitted checkpoint.
        return {"schema": FULL_SCHEMA, "config": asdict(self.config),
                "base": self.base.checkpoint_payload(), "computer": self.computer.state_dict(),
                "feedback": {name: getattr(self, name).state_dict() for name in (
                    "semantic_feedback", "query_feedback", "character_feedback")},
                "parameter_mode": self._parameter_mode, "admission": self.admission,
                "model_identity": self.model_identity(), "metadata": metadata}

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "FullFusionModel":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != FULL_SCHEMA:
            raise ValueError("checkpoint is not the optional v2 composed model")
        saved = payload["base"]
        if saved.get("schema") != BASE_SCHEMA:
            raise ValueError("nested base checkpoint has an incompatible schema")
        native = ConnectedCOGTinyLM(ConnectedConfig(**saved["native_config"]))
        base = FusionModel(native, saved["chars"], config=FusionConfig(**saved["config"]),
                           memory_enabled=saved["memory_enabled"], admission=saved["admission"])
        base.load_state_dict(saved["model"], strict=True)
        model = cls(base, config=FullConfig(**payload["config"]), admission=payload["admission"])
        model.computer.load_state_dict(payload["computer"], strict=True)
        for name in ("semantic_feedback", "query_feedback", "character_feedback"):
            getattr(model, name).load_state_dict(payload["feedback"][name], strict=True)
        mode = payload["parameter_mode"]
        if mode == "numeric_component_only":
            model.numeric_training_only()
        elif mode != "joint_trainable":
            raise ValueError("unknown component training mode")
        if model.model_identity() != payload["model_identity"]:
            raise ValueError("composed checkpoint weights or implementation identity differ")
        return model
