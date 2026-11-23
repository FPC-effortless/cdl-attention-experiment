"""Automatic crash-safe continuation of one immutable NanoCOG inference stream.

Decoder state is conversation-local. It is never promoted to reusable semantic
facts: those use the store's separate observation/proposal/record interface.
"""
import hashlib
import inspect
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules import module as torch_module

from .. import model as model_source
from .. import interaction as interaction_source
from ..comm import packet as packet_source
from ..comm.packet import PacketCommunication, PacketState
from ..interaction import PacketAndTypedProbe, PacketMixer
from ..model import DecodeState, Decoder, LocalOnly
from ..state import records as records_source
from ..state.records import RecordState, StableRecordMemory, TypedRecordBoard
from .codec import CodecError, CodecLimits, decode, encode, tensor_bytes
from .store import ConflictError, IntegrityError, ReceiptExpiredError


# The supported implementation is fixed when this trusted adapter is imported.
# Disk source hashes alone cannot detect a later class-level monkeypatch. Keep
# execution references as well; this is not protection against code controlling
# the interpreter before import or modifying the validator itself.
_EXECUTION_METHODS = {
    Decoder: ("forward",), LocalOnly: ("forward",),
    PacketMixer: ("forward",), PacketAndTypedProbe: ("forward",),
    PacketCommunication: ("forward", "linear", "_pool", "_core", "_project", "_validate", "initial_state"),
    StableRecordMemory: ("forward",),
    TypedRecordBoard: ("apply", "read", "empty", "_shape", "validate_state"),
    nn.Linear: ("forward",), nn.Conv1d: ("forward", "_conv_forward"), nn.Embedding: ("forward",),
}
_EXECUTION_BINDINGS = tuple(
    (owner, name, getattr(owner, name)) for owner, names in _EXECUTION_METHODS.items() for name in names
) + tuple((nn.Module, name, getattr(nn.Module, name)) for name in
          ("__call__", "_call_impl", "_wrapped_call_impl")) + ((model_source, "norm", model_source.norm),)


@dataclass(frozen=True)
class DecodeResult:
    logits: torch.Tensor
    revision: int
    tokens_seen: int
    replayed: bool


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _checked_name(value, label):
    if type(value) is not str or not value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be a nonempty identifier of at most 512 characters")
    return value


def _model_identity(model, tokenizer_id, session_id):
    if type(model) is not Decoder:
        raise TypeError("this adapter accepts the reference Decoder only")
    if any(getattr(owner, name) is not original for owner, name, original in _EXECUTION_BINDINGS):
        raise ValueError("patched reference execution methods are unsupported")
    if any(m.training for m in model.modules()):
        raise ValueError("persistent decoding requires model.eval()")
    if torch_module._global_forward_hooks or torch_module._global_forward_pre_hooks:
        raise ValueError("global inference hooks are unsupported")
    if any(any(name in m.__dict__ for name in
                   (*_EXECUTION_METHODS.get(type(m), ()), "_call_impl", "_wrapped_call_impl"))
           or m._forward_hooks or m._forward_pre_hooks
           or m._compiled_call_impl is not None
           for m in model.modules()):
        raise ValueError("patched forwards, compiled calls and inference hooks are unsupported")
    if any(type(m) not in (LocalOnly, PacketMixer, PacketAndTypedProbe) for m in model.mixers):
        raise TypeError("unsupported decoder mixer")
    supported_modules = (Decoder, LocalOnly, PacketMixer, PacketAndTypedProbe,
                         PacketCommunication, StableRecordMemory, TypedRecordBoard,
                         nn.ModuleList, nn.Linear, nn.Conv1d, nn.Embedding)
    if any(type(m) not in supported_modules for m in model.modules()):
        raise TypeError("unsupported decoder submodule")
    params = tuple(model.parameters())
    device, dtype = model.embedding.weight.device, model.embedding.weight.dtype
    if any(p.device != device or p.dtype != dtype for p in params):
        raise ValueError("persistent decoding requires one model device and floating dtype")
    if dtype not in (torch.float32, torch.float64):
        # The reference packet/typed composition is not a qualified mixed-precision backend.
        raise ValueError("persistent reference decoding supports float32 or float64")
    state_hash = hashlib.sha256()
    schema_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device != device:
            raise ValueError("model parameters and buffers must share one device")
        header = _json_bytes([name, str(tensor.dtype), list(tensor.shape)])
        raw = tensor_bytes(tensor)
        state_hash.update(header + b"\0" + raw)
        if name.endswith(".permits") or name.endswith(".links"):
            schema_hash.update(header + b"\0" + raw)
    # Include non-tensor geometry and implementation bytes, not just compatible shapes.
    geometry = []
    # Matching parameter shapes is insufficient: e.g. stride/dilation/padding
    # can change a convolution while retaining the same output geometry. Bind
    # every execution option of the supported reference modules. This is not a
    # sandbox against arbitrary mutation of Python class/function definitions.
    attributes = ("width", "lookback", "groups", "topics", "packet_width", "capacity", "local_window",
                  "in_features", "out_features", "in_channels", "out_channels", "kernel_size",
                  "stride", "padding", "dilation", "padding_mode", "transposed", "output_padding",
                  "_reversed_padding_repeated_twice", "num_embeddings", "embedding_dim",
                  "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse")
    for name, module in model.named_modules():
        geometry.append([name, type(module).__module__, type(module).__qualname__,
                         {a: getattr(module, a) for a in attributes if hasattr(module, a)}])
    sources = {}
    for module in (model_source, interaction_source, packet_source, records_source):
        path = inspect.getsourcefile(module)
        if path is None:
            raise ValueError("decoder implementation source is unavailable for fingerprinting")
        sources[module.__name__] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config = asdict(model.cfg)
    identity = {"weights_sha256": state_hash.hexdigest(), "geometry": geometry,
                "config": config, "implementation": sources}
    return {"format": "nanocog-persistent-decoder", "version": 2,
            "model_sha256": hashlib.sha256(_json_bytes(identity)).hexdigest(),
            "schema_sha256": schema_hash.hexdigest(), "tokenizer_id": tokenizer_id,
            "session_id": session_id, "batch_size": 1, "config": config,
            "runtime": {"torch": str(torch.__version__), "device": str(device), "dtype": str(dtype)}}


def _tensor(tensor, shape, dtype, device):
    if (type(tensor) is not torch.Tensor or tensor.shape != shape or
            tensor.dtype != dtype or tensor.device != device):
        raise IntegrityError("persisted tensor shape, dtype, or device mismatch")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise IntegrityError("nonfinite persisted tensor")


def _packet(state, module, seen, device, dtype):
    if type(state) is not PacketState or type(module) is not PacketCommunication:
        raise IntegrityError("invalid persisted packet state")
    _tensor(state.local, (1, module.local_window - 1, module.width), dtype, device)
    _tensor(state.packets, (1, module.groups, module.topics, module.capacity, module.packet_width), dtype, device)
    _tensor(state.present, (1, module.groups, module.topics, module.capacity), dtype, device)
    _tensor(state.positions, (1,), torch.long, device)
    if (state.positions != seen).any().item():
        raise IntegrityError("packet positions disagree with decoder chronology")
    if ((state.present != 0) & (state.present != 1)).any().item():
        raise IntegrityError("packet occupancy must be binary")
    if (state.present[..., 1:] > state.present[..., :-1]).any().item():
        raise IntegrityError("packet occupancy contains a history gap")
    if (state.present.sum(dim=(-2, -1)) > seen).any().item():
        raise IntegrityError("packet occupancy exceeds consumed events")
    if (state.packets.masked_select((state.present == 0).unsqueeze(-1)) != 0).any().item():
        raise IntegrityError("unoccupied packet carries a payload")
    if seen < module.local_window - 1 and (state.local[:, seen:] != 0).any().item():
        raise IntegrityError("packet local window contains events before stream start")


def validate_decode_state(state, model):
    """Check the complete supported state, beyond the model's hot-loop checks."""
    if (type(state) is not DecodeState or type(state.tokens_seen) is not int or
            not 0 < state.tokens_seen < 2**63 or type(state.mixer_states) is not tuple or
            len(state.mixer_states) != model.cfg.layers):
        raise IntegrityError("invalid decoder state or chronology")
    device, dtype = model.embedding.weight.device, model.embedding.weight.dtype
    seen = state.tokens_seen
    _tensor(state.previous_embedding, (1, 1, model.cfg.width), dtype, device)
    for module, current in zip(model.mixers, state.mixer_states):
        if type(module) is LocalOnly:
            _tensor(current, (1, module.lookback - 1, module.width), dtype, device)
            if seen < module.lookback - 1 and (current[:, :module.lookback - 1 - seen] != 0).any().item():
                raise IntegrityError("local state contains events before stream start")
        elif type(module) is PacketMixer:
            _packet(current, module.packets, seen, device, dtype)
        elif type(module) is PacketAndTypedProbe:
            if type(current) is not tuple or len(current) != 2 or type(current[1]) is not RecordState:
                raise IntegrityError("invalid packet/record state pair")
            _packet(current[0], module.packets, seen, device, dtype)
            records = module.records.board.records
            for field, shape in zip(current[1]._parts(), ((1, records), (1, records), (1,), (1, records))):
                _tensor(field, shape, torch.long, device)
            try:
                module.records.board.validate_state(current[1])
            except (TypeError, ValueError, IndexError, RuntimeError) as exc:
                raise IntegrityError("invalid persisted typed record state") from exc
            if current[1].clock.shape != (1,) or (current[1].clock != seen).any().item():
                raise IntegrityError("record clock disagrees with decoder chronology")
        else:
            raise IntegrityError("unsupported mixer state")
    return state


class PersistentDecoder:
    """Attach a single stream; every successful run is persisted before return.

    The caller supplies a stable tokenizer identity (prefer an asset digest), an
    already-scoped MemoryStore, and immutable eval-mode weights. New session IDs
    start new transient streams. Model/tokenizer upgrades require a new session;
    reusable facts in the store are unaffected. This wrapper owns no model fit.

    Batches, autocast, arbitrary mixers, modified forward hooks and cross-runtime
    state reuse are rejected rather than silently approximated. Float results
    retain the existing reference's tolerance limits across chunk shapes.
    """
    def __init__(self, model, store, *, session_id, tokenizer_id,
                 max_tokens_per_run=2048, codec_limits=None):
        self.model, self.store = model, store
        self.session_id = _checked_name(session_id, "session_id")
        self.tokenizer_id = _checked_name(tokenizer_id, "tokenizer_id")
        if type(max_tokens_per_run) is not int or max_tokens_per_run < 1:
            raise ValueError("max_tokens_per_run must be a positive integer")
        self.max_tokens_per_run = max_tokens_per_run
        self.codec_limits = codec_limits or CodecLimits()
        self._manifest = _model_identity(model, self.tokenizer_id, self.session_id)
        self._lock = threading.RLock()
        existing = store.load_checkpoint(self.session_id)
        if existing is not None:
            self._check_manifest(existing["manifest"])
            self._load_state(existing["payload"])

    def _check_manifest(self, manifest):
        if manifest != self._manifest:
            raise IntegrityError("decoder checkpoint model, tokenizer, schema, runtime, or session mismatch")

    def _check_model(self):
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("persistent reference decoding does not support autocast")
        if _model_identity(self.model, self.tokenizer_id, self.session_id) != self._manifest:
            raise IntegrityError("model changed after persistent session attachment")

    def _load_state(self, payload):
        if type(payload) is not dict or set(payload) != {"state"}:
            raise IntegrityError("malformed decoder checkpoint payload")
        try:
            state = decode(payload["state"], device=self.model.embedding.weight.device, limits=self.codec_limits)
        except (CodecError, RuntimeError, OverflowError) as exc:
            raise IntegrityError("cannot decode persisted state") from exc
        return validate_decode_state(state, self.model)

    def _replay(self, receipt, request_hash, input_length):
        self._check_manifest(receipt["manifest"])
        if receipt["request_hash"] != request_hash:
            raise ConflictError("request ID already belongs to different decoder input")
        saved = receipt["result"]
        if saved is None:
            raise ReceiptExpiredError("request was committed but its cached decoder output has expired")
        if (type(saved) is not dict or set(saved) != {"logits", "tokens_seen", "input_length"}
                or type(saved["tokens_seen"]) is not int or not input_length <= saved["tokens_seen"] < 2**63
                or type(saved["input_length"]) is not int or saved["input_length"] != input_length
                or type(receipt["revision"]) is not int or receipt["revision"] < 1):
            raise IntegrityError("invalid decoder receipt")
        try:
            logits = decode(saved["logits"], device=self.model.embedding.weight.device, limits=self.codec_limits)
        except (CodecError, RuntimeError, OverflowError) as exc:
            raise IntegrityError("cannot decode persisted output") from exc
        _tensor(logits, (1, input_length, self.model.cfg.vocab_size), torch.float32,
                self.model.embedding.weight.device)
        return DecodeResult(logits, receipt["revision"], saved["tokens_seen"], True)

    def run(self, tokens, *, request_id):
        """Reload, infer and atomically commit, or return an exact retry receipt.

        Concurrent sessions use database compare-and-swap. A losing stale writer
        raises ConflictError and must be retried deliberately with current state.
        No output is returned as successful before its state transaction commits.

        New inference linearizes at its commit; an exact retry linearizes at its
        receipt read. Erasure invalidates later reads and rejects in-flight new
        commits, including the first commit. It cannot revoke an output already
        read by an overlapping retry before that erasure committed.
        """
        with self._lock:
            request_id = _checked_name(request_id, "request_id")
            erasure_generation = self.store.erasure_generation()
            self._check_model()
            if (type(tokens) is not torch.Tensor or tokens.layout != torch.strided
                    or tokens.dtype != torch.long or tokens.ndim != 2
                    or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= self.max_tokens_per_run
                    or tokens.device != self.model.embedding.weight.device):
                raise ValueError("tokens must be bounded int64 [1,T] on the model device")
            if ((tokens < 0) | (tokens >= self.model.cfg.vocab_size)).any().item():
                raise ValueError("tokens outside model vocabulary")
            input_length = tokens.shape[1]
            if input_length * self.model.cfg.vocab_size * 4 > self.codec_limits.max_bytes:
                raise CodecError("decoder output exceeds codec byte limit; use smaller chunks")
            # Own input storage so a caller cannot mutate it between hashing and inference.
            tokens = tokens.detach().clone()
            request_hash = hashlib.sha256(_json_bytes(self._manifest) + b"\0" +
                                          _json_bytes(list(tokens.shape)) + b"\0" + tensor_bytes(tokens)).hexdigest()
            prior = self.store.checkpoint_receipt(self.session_id, request_id)
            if prior is not None:
                return self._replay(prior, request_hash, input_length)
            current = self.store.load_checkpoint(self.session_id)
            state, revision = None, 0
            if current is not None:
                self._check_manifest(current["manifest"])
                state = self._load_state(current["payload"])
                revision = current["revision"]
            before = 0 if state is None else state.tokens_seen
            if before + input_length >= 2**63:
                raise IntegrityError("decoder token clock would overflow")
            with torch.inference_mode():
                logits, updated = self.model(tokens, state)
            self._check_model()
            validate_decode_state(updated, self.model)
            if updated.tokens_seen != before + input_length:
                raise IntegrityError("decoder continuation did not consume exactly the input tokens")
            _tensor(logits, (1, input_length, self.model.cfg.vocab_size), torch.float32,
                    self.model.embedding.weight.device)
            payload = {"state": encode(updated, limits=self.codec_limits)}
            result = {"logits": encode(logits, limits=self.codec_limits),
                      "tokens_seen": updated.tokens_seen, "input_length": input_length}
            receipt = self.store.commit_checkpoint(
                self.session_id, request_id=request_id, request_hash=request_hash,
                expected_revision=revision, manifest=self._manifest, payload=payload, result=result,
                expected_erasure_generation=erasure_generation)
            # A simultaneous identical request can win the transaction. Always
            # return the stored receipt, whose result is authoritative for retries.
            restored = self._replay(receipt, request_hash, input_length)
            return DecodeResult(restored.logits, restored.revision, restored.tokens_seen,
                                bool(receipt.get("duplicate", False)))
