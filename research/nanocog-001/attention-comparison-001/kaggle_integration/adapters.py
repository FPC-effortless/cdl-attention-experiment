"""Source-only, FP32 engineering adapter for pinned nanoCOG and nanochat.

Set NANOCHAT_DTYPE=float32 *before importing this module*. Put the pinned
nanochat checkout and research/nanocog-001 on PYTHONPATH. This module neither
downloads assets nor creates an optimizer, tokenizer, training job, or dataset.
The imported upstream attention package may perform its own kernel discovery;
the launcher must control and record that dependency boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import inspect
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig
from nanocog.comm.packet import PacketCommunication, PacketState
from nanocog.interaction import PacketMixer
from nanocog.model import Config, DecodeState, Decoder, LocalOnly


UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
CANDIDATE_SOURCE_COMMIT = "39be434ff160d4e6507df8d2114979ba5529caff"
COMPARISON_SOURCE_COMMIT = "1a91168b216a49543cc839153fa299a262320730"
STATE_SCHEMA = "nanocog-kaggle-aligned-fp32-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CommonGeometry:
    vocab_size: int = 32768
    width: int = 128
    layers: int = 2
    n_head: int = 2
    n_kv_head: int = 1
    max_seq_len: int = 256

    def validate(self) -> None:
        _require(all(type(v) is int and v > 0 for v in asdict(self).values()),
                 "geometry must consist of positive integers")
        _require(self.width >= 24, "smear gate requires width >= 24")
        _require(self.width % self.n_head == 0, "width must divide into query heads")
        _require(self.n_head > self.n_kv_head and self.n_head % self.n_kv_head == 0,
                 "this baseline requires genuine GQA: n_head > n_kv_head and divisible")
        _require((self.width // self.n_head) % 2 == 0, "rotary head dimension must be even")


def shared_tensor_mapping(layers: int) -> dict[str, str]:
    """Candidate parameter name -> exact upstream common-shell parameter name."""
    result = {
        "embedding.weight": "transformer.wte.weight",
        "output.weight": "lm_head.weight",
        "resid": "resid_lambdas",
        "x0_gain": "x0_lambdas",
        "smear_gate.weight": "smear_gate.weight",
        "smear_gain": "smear_lambda",
        "backout": "backout_lambda",
    }
    for i in range(layers):
        result[f"mlps.{i}.0.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
        result[f"mlps.{i}.1.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"
    return result


@torch.no_grad()
def copy_common_shell(source: GPT, destination: Decoder) -> None:
    src, dst = dict(source.named_parameters()), dict(destination.named_parameters())
    for candidate_name, upstream_name in shared_tensor_mapping(destination.cfg.layers).items():
        a, b = src[upstream_name], dst[candidate_name]
        if candidate_name in ("smear_gain", "backout"):
            _require(a.numel() == b.numel() == 1, "mapped scalar must have exactly one element")
            a = a.reshape(b.shape)  # upstream [1], original candidate []; values are identical
        _require(a.shape == b.shape and a.dtype == b.dtype, "common-shell shape/dtype mismatch")
        b.copy_(a)


class DecodeSession:
    """Opaque mutable owned session; direct writes to its internals are unsupported.

    Only an adapter may consume this object. An explicit session_id is required
    at each boundary to catch accidental exchange of live request state. State
    serialization and concurrent calls on the same session are not implemented.
    """
    __slots__ = ("_owner", "_stamp", "_schema", "session_id", "batch_size",
                 "max_tokens", "position", "_payload", "_failed")

    def __init__(self, owner: object, stamp: tuple, session_id: str,
                 batch_size: int, max_tokens: int) -> None:
        self._owner, self._stamp = owner, stamp
        self._schema = STATE_SCHEMA
        self.session_id, self.batch_size, self.max_tokens = session_id, batch_size, max_tokens
        self.position = 0
        self._payload: DecodeState | KVCache | None = None
        self._failed = False


class _OwnedPacketCall:
    """Only the mixer call changes; the enclosing Decoder.forward is reused."""
    def __init__(self, mixer: PacketMixer) -> None:
        self.layer = mixer.packets

    def __call__(self, x: Tensor, state: PacketState) -> tuple[Tensor, PacketState]:
        _require(state is not None, "owned packet call requires private initialized state")
        return self.layer.step_inplace_(x, state), state


class _OwnedDecoderView:
    """Borrow source attributes without registering or cloning model parameters.

    Decoder.forward at the pinned source only reads these attributes. Delegation
    avoids copying its residual/smear/MLP/head implementation. No source class,
    module, or instance forward method is monkey-patched.
    """
    def __init__(self, source: Decoder) -> None:
        self.source = source
        self.mixers = [_OwnedPacketCall(m) if isinstance(m, PacketMixer) else m
                       for m in source.mixers]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def _storage_totals(tensors: list[tuple[str, Tensor]]) -> tuple[int, int]:
    logical, allocated, seen = 0, 0, set()
    for _, tensor in tensors:
        logical += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            allocated += storage.nbytes()
    return logical, allocated


def _metadata_size(value: Any, seen: set[int] | None = None) -> int:
    """Recursive Python-object estimate, excluding Tensor objects and storage."""
    seen = set() if seen is None else seen
    if isinstance(value, Tensor) or id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        return total + sum(_metadata_size(k, seen) + _metadata_size(v, seen)
                           for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return total + sum(_metadata_size(v, seen) for v in value)
    if isinstance(value, DecodeSession):
        return total + sum(_metadata_size(getattr(value, key), seen) for key in value.__slots__)
    if is_dataclass(value) and not isinstance(value, type):
        # Include the instance dict as well as its field values.
        return total + _metadata_size(vars(value), seen)
    if isinstance(value, KVCache):
        return total + _metadata_size(vars(value), seen)
    return total


class ModelAdapter(nn.Module):
    """Common loss interface plus explicit, aligned, owned serving sessions."""
    def __init__(self, kind: str, model: nn.Module, geometry: CommonGeometry) -> None:
        super().__init__()
        _require(kind in ("packet", "local", "gqa"), "unknown backend")
        self.kind, self.model, self.geometry = kind, model, geometry
        self._owner = object()
        self._owned_view = None if kind == "gqa" else _OwnedDecoderView(model)

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _stamp(self) -> tuple:
        # Bind buffers too: changing rotary state changes cached model semantics.
        # These models are constructed outside inference_mode so version counters
        # exist even when this check subsequently runs under inference_mode.
        tensors = list(self.model.named_parameters()) + list(self.model.named_buffers())
        tensor_stamp = tuple((name, id(p), p._version, p.data_ptr(), str(p.device), str(p.dtype), tuple(p.shape))
                             for name, p in tensors)
        config = self.model.config if self.kind == "gqa" else self.model.cfg
        windows = tuple(self.model.window_sizes) if self.kind == "gqa" else ()
        return tensor_stamp, tuple(sorted(asdict(config).items())), windows

    def _tokens(self, tokens: Tensor, *, batch: int | None = None,
                valid_mask: Tensor | None = None) -> None:
        _require(isinstance(tokens, Tensor) and tokens.ndim == 2 and tokens.dtype == torch.long,
                 "tokens must be a [B,T] int64 tensor")
        _require(min(tokens.shape) > 0, "empty batches/chunks are unsupported")
        _require(tokens.device == self.get_device(), "tokens must be on the model device")
        _require(batch is None or tokens.shape[0] == batch, "session batch size differs")
        _require(tokens.shape[1] <= self.geometry.max_seq_len, "sequence exceeds declared context")
        _require(not bool(((tokens < 0) | (tokens >= self.geometry.vocab_size)).any()),
                 "token outside declared vocabulary")
        if valid_mask is not None:
            _require(valid_mask.shape == tokens.shape and valid_mask.dtype == torch.bool
                     and valid_mask.device == tokens.device, "invalid validity mask")
            _require(bool(valid_mask.all()), "padding/mixed row validity is unsupported")

    def forward(self, tokens: Tensor, targets: Tensor | None = None,
                loss_reduction: str = "mean") -> Tensor:
        """Stateless independent batch. Never carries serving state into training."""
        self._tokens(tokens)
        _require(tokens.shape[1] > 1, "common stateless interface requires T > 1; use prefill for T=1")
        _require(loss_reduction in ("none", "mean", "sum"), "invalid reduction")
        if targets is not None:
            _require(targets.shape == tokens.shape and targets.dtype == torch.long
                     and targets.device == tokens.device, "target geometry/device/dtype mismatch")
            _require(not bool(((targets < -1) | (targets >= self.geometry.vocab_size)).any()),
                     "target outside vocabulary or ignore_index=-1")
            _require(loss_reduction != "mean" or bool(targets.ne(-1).any()),
                     "mean loss requires at least one observed target")
            # Upstream GPT uses targets.view(-1), whereas Decoder uses reshape.
            # Normalize symmetrically so ordinary next-token slices work for B>1.
            # This copy, when needed, is part of the adapter's measured work.
            targets = targets.contiguous()
        if self.kind == "gqa":
            return self.model(tokens, targets=targets, loss_reduction=loss_reduction)
        return self.model(tokens, state=None, targets=targets, loss_reduction=loss_reduction)[0]

    def new_session(self, *, batch_size: int, max_tokens: int, session_id: str) -> DecodeSession:
        _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
        _require(type(max_tokens) is int and 0 < max_tokens <= self.geometry.max_seq_len,
                 "session capacity must be positive and within declared full context")
        _require(type(session_id) is str and bool(session_id), "session_id must be nonempty")
        return DecodeSession(self._owner, self._stamp(), session_id, batch_size, max_tokens)

    def _check(self, state: DecodeSession, session_id: str) -> None:
        _require(isinstance(state, DecodeSession) and state._owner is self._owner,
                 "state belongs to another model/adapter")
        _require(state._schema == STATE_SCHEMA and state.session_id == session_id,
                 "session identifier/schema mismatch")
        _require(not state._failed, "session was invalidated by a failed owned update; create a new session")
        _require(state._stamp == self._stamp(), "model parameters/buffers/config changed after session creation")
        _require(not self.training and not self.model.training, "serving requires eval mode")
        _require(0 <= state.position <= state.max_tokens <= self.geometry.max_seq_len,
                 "invalid session position/capacity")
        payload = state._payload
        if payload is None:
            _require(state.position == 0, "missing initialized state")
        elif self.kind == "gqa":
            _require(isinstance(payload, KVCache), "incorrect cache type")
            _require(payload.batch_size == state.batch_size
                     and bool(payload.cache_seqlens.eq(state.position).all()),
                     "mixed row positions are unsupported")
        else:
            _require(isinstance(payload, DecodeState) and payload.tokens_seen == state.position,
                     "decoder position mismatch")
            _require(payload.previous_embedding.shape == (state.batch_size, 1, self.geometry.width),
                     "preceding embedding geometry mismatch")
            for layer in payload.mixer_states:
                if isinstance(layer, PacketState):
                    _require(bool(layer.positions.eq(state.position).all()),
                             "mixed packet row positions are unsupported")

    def _kv_cache(self, state: DecodeSession) -> KVCache:
        return KVCache(batch_size=state.batch_size, num_heads=self.geometry.n_kv_head,
                       seq_len=state.max_tokens, head_dim=self.geometry.width // self.geometry.n_head,
                       num_layers=self.geometry.layers, device=self.get_device(), dtype=torch.float32)

    @torch.inference_mode()
    def prefill(self, tokens: Tensor, state: DecodeSession, *, session_id: str,
                valid_mask: Tensor | None = None) -> Tensor:
        self._check(state, session_id)
        self._tokens(tokens, batch=state.batch_size, valid_mask=valid_mask)
        _require(state.position == 0 and state._payload is None, "prefill requires a reset/empty session")
        _require(tokens.shape[1] <= state.max_tokens, "prefill exceeds session capacity")
        if self.kind == "gqa":
            payload = self._kv_cache(state)
            logits = self.model(tokens, kv_cache=payload)
        else:
            logits, fresh = self.model(tokens, state=None)
            # Independent retained storage is an explicit ownership boundary.
            layers = tuple(s.detached() if isinstance(s, PacketState) else s.detach().clone()
                           for s in fresh.mixer_states)
            payload = DecodeState(layers, fresh.previous_embedding.detach().clone(), fresh.tokens_seen)
        state._payload = payload
        state.position = tokens.shape[1]
        self._check(state, session_id)
        return logits

    @torch.inference_mode()
    def step(self, tokens: Tensor, state: DecodeSession, *, session_id: str,
             valid_mask: Tensor | None = None) -> Tensor:
        self._check(state, session_id)
        self._tokens(tokens, batch=state.batch_size, valid_mask=valid_mask)
        _require(tokens.shape[1] == 1, "cached continuation accepts exactly one token per row")
        _require(state.position > 0 and state._payload is not None, "step requires a prefilled session")
        _require(state.position < state.max_tokens, "session capacity exhausted")
        try:
            if self.kind == "gqa":
                logits = self.model(tokens, kv_cache=state._payload)
            else:
                # Invoke the exact original shell; only PacketMixer calls use the
                # source-owned update. LocalOnly retains its native functional call.
                logits, payload = Decoder.forward(self._owned_view, tokens, state=state._payload)
                state._payload = payload
            state.position += 1
            self._check(state, session_id)
        except Exception:
            # An in-place kernel can fail after mutating only part of its cache.
            # Never present that partially updated state as a reusable prefix.
            state._failed = True
            raise
        return logits

    @torch.inference_mode()
    def fork(self, state: DecodeSession, rows: Tensor, *, session_id: str,
             new_session_id: str) -> DecodeSession:
        self._check(state, session_id)
        _require(new_session_id != session_id, "fork must have a different session identifier")
        _require(isinstance(rows, Tensor) and rows.ndim == 1 and rows.dtype == torch.long
                 and rows.device == self.get_device() and rows.numel() > 0,
                 "fork rows must be nonempty same-device int64 vector")
        _require(not bool(((rows < 0) | (rows >= state.batch_size)).any()), "fork row out of bounds")
        child = self.new_session(batch_size=rows.numel(), max_tokens=state.max_tokens,
                                 session_id=new_session_id)
        if state._payload is None:
            return child
        old = state._payload
        if self.kind == "gqa":
            payload = self._kv_cache(child)
            p = state.position
            payload.k_cache[:, :, :p].copy_(old.k_cache[:, :, :p].index_select(1, rows))
            payload.v_cache[:, :, :p].copy_(old.v_cache[:, :, :p].index_select(1, rows))
            payload.cache_seqlens.copy_(old.cache_seqlens.index_select(0, rows))
            payload.prev_embedding = None if old.prev_embedding is None else old.prev_embedding.index_select(0, rows)
        else:
            payload = DecodeState(tuple(s.index_select(rows) if isinstance(s, PacketState)
                                        else s.index_select(0, rows) for s in old.mixer_states),
                                  old.previous_embedding.index_select(0, rows), old.tokens_seen)
        child._payload, child.position = payload, state.position
        self._check(child, new_session_id)
        return child

    @torch.inference_mode()
    def reset(self, state: DecodeSession, *, session_id: str) -> None:
        """Release whole-session cache; partial-row resets are unsupported."""
        self._check(state, session_id)
        state._payload, state.position = None, 0

    def state_tensors(self, state: DecodeSession, *, session_id: str) -> list[tuple[str, Tensor]]:
        """Read-only inventory; callers must not mutate returned tensors."""
        self._check(state, session_id)
        payload = state._payload
        if payload is None:
            return []
        if self.kind == "gqa":
            result = [("kv.k_cache", payload.k_cache), ("kv.v_cache", payload.v_cache),
                      ("kv.cache_seqlens", payload.cache_seqlens)]
            if payload.prev_embedding is not None:
                result.append(("kv.previous_embedding", payload.prev_embedding))
            return result
        result = [("decoder.previous_embedding", payload.previous_embedding)]
        for i, layer in enumerate(payload.mixer_states):
            if isinstance(layer, PacketState):
                result.extend((f"layer.{i}.{name}", getattr(layer, name))
                              for name in ("local", "packets", "present", "positions"))
            else:
                result.append((f"layer.{i}.local", layer))
        return result

    def state_report(self, state: DecodeSession, *, session_id: str) -> dict[str, Any]:
        tensors = self.state_tensors(state, session_id=session_id)
        logical, allocated = _storage_totals(tensors)
        wrappers = sum(sys.getsizeof(t) for t in {id(t): t for _, t in tensors}.values())
        metadata = _metadata_size(state)
        return {
            "schema": STATE_SCHEMA, "backend": self.kind, "batch_size": state.batch_size,
            "position": state.position, "declared_capacity_tokens": state.max_tokens,
            "tensor_logical_bytes": logical, "tensor_storage_bytes": allocated,
            "python_tensor_wrapper_bytes_estimate": wrappers,
            "python_metadata_bytes_estimate": metadata,
            "retained_state_bytes_estimate": allocated + wrappers + metadata,
            "tensors": [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
                         "logical_bytes": t.numel() * t.element_size(),
                         "retained_storage_bytes": t.untyped_storage().nbytes()}
                        for name, t in tensors],
            "accounting": "Underlying tensor storages deduplicated, including allocated unused KV capacity and retained views. Python metadata uses recursive sys.getsizeof, including position, identity and model signature.",
            "outside_retained_state": ["model parameters/buffers (separate parameter report)",
                                       "returned logits", "temporary and fork/prefill copies",
                                       "allocator reserve/fragmentation", "peak process/device memory"],
        }

    def parameter_report(self) -> dict[str, Any]:
        mapping = shared_tensor_mapping(self.geometry.layers)
        common_names = set(mapping.values()) if self.kind == "gqa" else set(mapping)
        inverse = {v: k for k, v in mapping.items()}
        entries = []
        for name, p in self.model.named_parameters():
            upstream_name = name if self.kind == "gqa" else mapping.get(name)
            canonical = inverse.get(name) if self.kind == "gqa" else name
            if name in common_names:
                if canonical == "embedding.weight":
                    role, family = "common_embedding", "upstream AdamW embedding group"
                elif canonical == "output.weight":
                    role, family = "common_output", "upstream AdamW unembedding group"
                elif canonical == "resid":
                    role, family = "common_residual", "upstream AdamW residual group"
                elif canonical == "x0_gain":
                    role, family = "common_x0", "upstream AdamW x0 group"
                elif canonical in ("smear_gate.weight", "smear_gain", "backout"):
                    role, family = "common_smear_backout", "upstream AdamW smear group"
                else:
                    role, family = "common_mlp_matrix", "upstream Muon matrix group"
            elif self.kind == "gqa":
                if name.startswith("value_embeds."):
                    role, family = "attention_value_embedding", "upstream AdamW value-embedding group"
                else:
                    role, family = "attention_matrix", "upstream Muon matrix group"
            else:
                if name.endswith(".bias"):
                    role = "candidate_bias"
                elif name.endswith("local_age") or name.endswith("packet_age"):
                    role = "candidate_age_weight"
                elif p.ndim == 3:
                    role = "candidate_depthwise_convolution"
                else:
                    role = "candidate_matrix"
                family = "UNADMITTED: candidate-specific optimizer group required"
            entries.append({"name": name, "shape": list(p.shape), "numel": p.numel(),
                            "trainable": p.requires_grad, "dtype": str(p.dtype), "role": role,
                            "common_shell": name in common_names,
                            "upstream_parameter": upstream_name, "optimizer_reference": family})
        params = list(self.model.named_parameters())
        buffers = list(self.model.named_buffers())
        return {
            "backend": self.kind,
            "total_unique_parameters": sum(p.numel() for _, p in params),
            "trainable_parameters": sum(p.numel() for _, p in params if p.requires_grad),
            "common_shell_parameters": sum(e["numel"] for e in entries if e["common_shell"]),
            "backend_specific_parameters": sum(e["numel"] for e in entries if not e["common_shell"]),
            "parameter_storage_bytes": _storage_totals(params)[1],
            "buffer_storage_bytes": _storage_totals(buffers)[1],
            "entries": entries, "optimizer_admission": "PENDING_NO_OPTIMIZER_CREATED",
            "active_gradient_or_quality_claim": False,
            "capacity_matched": False,
        }


def _source_receipt(cls: type) -> dict[str, str]:
    path = Path(inspect.getfile(cls)).resolve()
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_model_triplet(*, vocab_size: int = 32768, width: int = 128, layers: int = 2,
                        n_head: int = 2, n_kv_head: int = 1, max_seq_len: int = 256,
                        seed: int = 1729, device: str | torch.device = "cpu"
                        ) -> tuple[dict[str, ModelAdapter], dict[str, Any]]:
    """Initialize common-shell-matched models; neither trains nor qualifies them.

    LocalOnly is a separately parameterized no-memory control, ready for its own
    future training. It is not a packet-disabled checkpoint. No arm is trained.
    """
    geometry = CommonGeometry(vocab_size, width, layers, n_head, n_kv_head, max_seq_len)
    geometry.validate()
    _require(not torch.is_inference_mode_enabled(),
             "construct models outside inference_mode so parameter/buffer versions can bind sessions")
    _require(COMPUTE_DTYPE == torch.float32,
             "first adapter requires NANOCHAT_DTYPE=float32 before importing nanochat")
    _require(type(seed) is int and 0 <= seed < 2**63 - 3, "invalid initialization seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        baseline = GPT(GPTConfig(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=layers,
                                 n_head=n_head, n_kv_head=n_kv_head, n_embd=width, window_pattern="L"))
        baseline.init_weights()
        torch.random.default_generator.manual_seed(seed + 1)
        packet = Decoder(Config(vocab_size, width, layers), make_mixer=PacketMixer)
        torch.random.default_generator.manual_seed(seed + 2)
        local = Decoder(Config(vocab_size, width, layers), make_mixer=LocalOnly)
        copy_common_shell(baseline, packet)
        copy_common_shell(baseline, local)
    native = {"packet": packet, "local": local, "gqa": baseline}
    models = {kind: ModelAdapter(kind, model.to(device=device), geometry).eval()
              for kind, model in native.items()}
    _require(all(p.dtype == torch.float32 for m in models.values() for p in m.parameters()),
             "unexpected non-FP32 parameter")
    provenance = {
        "status": "UNTRAINED_ENGINEERING_MODELS_ONLY", "geometry": asdict(geometry),
        "initialization_seed": seed,
        "source_commit_declarations": {"nanochat": UPSTREAM_COMMIT,
                                       "nanocog": CANDIDATE_SOURCE_COMMIT,
                                       "comparison": COMPARISON_SOURCE_COMMIT},
        "source_files": {"GPT": _source_receipt(GPT), "KVCache": _source_receipt(KVCache),
                         "Decoder": _source_receipt(Decoder), "PacketMixer": _source_receipt(PacketMixer),
                         "PacketCommunication": _source_receipt(PacketCommunication)},
        "initialization": "Unchanged GPT.init_weights; exact independent copies of mapped common tensors into candidate/local. Backend-specific parameters retain their native initialization.",
        "shared_tensor_mapping": shared_tensor_mapping(layers),
        "shared_scalar_layout_conversion": {"smear_gain": "upstream [1] -> candidate []",
                                            "backout": "upstream [1] -> candidate []"},
        "common_shell_equality_required": True,
        "baseline": "Unchanged pinned GPT, explicit genuine GQA, window_pattern=L; attention value embeddings and rotary remain baseline-specific.",
        "dtype": "float32", "tokenizer_assets": None, "data_manifest": None,
        "optimizer_created": False, "training_updates": 0,
        "scientific_comparison_admitted": False,
        "parameter_reports": {kind: model.parameter_report() for kind, model in models.items()},
    }
    return models, provenance
