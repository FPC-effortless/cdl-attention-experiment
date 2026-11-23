"""Optional learned read-compute-write component and durable typed diagnostic.

Addresses and the write destination are supplied. This module does not infer
them from language. The numerical fit does not train the full model's language,
retrieval, or semantic-feedback projections. Derived records are typed envelopes:
only ``read_derived`` validates their parent dependencies before exposing a value.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .store import (ConflictError, Dependency, EvidencePacket, IntegrityError,
                    Key, Operation, Store, canonical)


COMPOSITION_SCHEMA = "cog-fusion-numeric-composition-v1"


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _save(value) -> bytes:
    stream = io.BytesIO(); torch.save(value, stream); return stream.getvalue()


def _load(raw: bytes):
    try:
        return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    except Exception as exc:
        raise IntegrityError("invalid numeric continuation") from exc


class StaleDerivedError(ConflictError):
    pass


class RecordComputer(nn.Module):
    """A real recurrent path from preceding reads to the proposed value."""
    def __init__(self, value_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        if type(value_dim) is not int or type(hidden_dim) is not int or min(value_dim, hidden_dim) < 1:
            raise ValueError("dimensions must be positive integers")
        self.value_dim, self.hidden_dim = value_dim, hidden_dim
        self.controller = nn.GRUCell(value_dim, hidden_dim)
        self.write_head = nn.Linear(hidden_dim, value_dim)

    def init_state(self, batch: int, device=None) -> Tensor:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        return torch.zeros(batch, self.hidden_dim, device=device or self.write_head.weight.device,
                           dtype=self.write_head.weight.dtype)

    def consume(self, value: Tensor, state: Tensor) -> Tensor:
        if (not isinstance(value, Tensor) or value.ndim != 2 or value.size(1) != self.value_dim
                or not isinstance(state, Tensor) or state.shape != (value.size(0), self.hidden_dim)):
            raise ValueError("numeric read/controller shape mismatch")
        if (value.device != self.write_head.weight.device or state.device != value.device
                or value.dtype != self.write_head.weight.dtype or state.dtype != value.dtype
                or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(state).all())):
            raise ValueError("numeric payload/controller must be finite and match model type")
        return self.controller(value, state)

    read = consume

    def propose(self, state: Tensor) -> Tensor:
        if (not isinstance(state, Tensor) or state.ndim != 2 or state.size(1) != self.hidden_dim
                or state.device != self.write_head.weight.device
                or state.dtype != self.write_head.weight.dtype or not bool(torch.isfinite(state).all())):
            raise ValueError("invalid numeric controller")
        return self.write_head(state)

    def forward(self, first: Tensor, second: Tensor, *, reset_between: bool = False) -> Tensor:
        state = self.consume(first, self.init_state(first.size(0), first.device))
        if reset_between:
            state = torch.zeros_like(state)
        return self.propose(self.consume(second, state))

    def model_identity(self) -> str:
        result = hashlib.sha256(canonical({"schema": COMPOSITION_SCHEMA, "value_dim": self.value_dim,
                                          "hidden_dim": self.hidden_dim, "source": source_hash()}).encode())
        for name, value in self.state_dict().items():
            result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    def checkpoint_payload(self, **metadata):
        return {"schema": COMPOSITION_SCHEMA, "value_dim": self.value_dim, "hidden_dim": self.hidden_dim,
                "source_sha256": source_hash(), "model": self.state_dict(),
                "model_identity": self.model_identity(), "metadata": metadata}

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RecordComputer":
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved.get("schema") != COMPOSITION_SCHEMA or saved.get("source_sha256") != source_hash():
            raise IntegrityError("numeric checkpoint schema or source identity differs")
        model = cls(saved["value_dim"], saved["hidden_dim"])
        model.load_state_dict(saved["model"], strict=True)
        if model.model_identity() != saved["model_identity"]:
            raise IntegrityError("numeric checkpoint weight identity differs")
        return model


def packet_from_dict(raw: dict) -> EvidencePacket:
    if not isinstance(raw, dict) or set(raw) != set(EvidencePacket.__dataclass_fields__):
        raise IntegrityError("derived dependency packet schema differs")
    try:
        values = dict(raw)
        values["key"] = tuple(values["key"])
        values["values"] = tuple(values["values"])
        values["pending"] = tuple(values["pending"])
        values["dependencies"] = tuple(Dependency(tuple(item["key"]), item["token"],
                                                  item["version"], item["status"])
                                       for item in values["dependencies"])
        return EvidencePacket(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("invalid derived dependency packet") from exc


class RecordRuntime:
    """Use the same Store to bind actual numeric reads, computation and write.

    ``model`` may be RecordComputer or FullFusionModel. The latter persists its
    complete base continuation and the shared record controller. A new compute
    operation resets only the record controller by default, matching training.

    Each parent packet and the initial snapshot must have one common revision.
    Commit uses that immutable revision under BEGIN IMMEDIATE. Every supported
    store mutation advances revision, so neither parent can change between the
    reads and publication. Destination aliases to either source are refused.
    """
    def __init__(self, model, path: str | Path, *, capacity: int | None = None):
        self.model = model.eval()
        self.computer = getattr(model, "computer", model)
        if not isinstance(self.computer, RecordComputer):
            raise TypeError("model must contain RecordComputer")
        self.full = model is not self.computer
        if self.full and not all(hasattr(model, name) for name in (
                "consume_selected", "propose_numeric_write", "clear_controller")):
            raise TypeError("full model lacks the explicit numeric interfaces")
        if self.computer.write_head.weight.device.type != "cpu":
            raise ValueError("durable composition is qualified on CPU")
        self.model_id = model.model_identity()
        self.store = Store(path, self.model_id, schema_id=COMPOSITION_SCHEMA, capacity=capacity)
        self.state = self._initial()
        self.pending_read = None
        self.operations = 0
        self.revision = 0
        self._restore(self.store.snapshot_state())

    def _initial(self):
        return self.model.init_state(1) if self.full else {"record_controller": self.computer.init_state(1)}

    def _controller(self, state=None):
        return (self.state if state is None else state)["record_controller"]

    def _restore(self, snapshot: dict):
        if snapshot["neural_state"]:
            saved = _load(snapshot["neural_state"])
            if (set(saved) != {"schema", "model_id", "state", "operations", "pending_read"}
                    or saved["schema"] != COMPOSITION_SCHEMA or saved["model_id"] != self.model_id
                    or type(saved["operations"]) is not int or saved["operations"] < 0):
                raise IntegrityError("numeric continuation identity differs")
            controller = saved["state"].get("record_controller")
            if (not isinstance(controller, Tensor) or controller.shape != (1, self.computer.hidden_dim)
                    or controller.dtype != self.computer.write_head.weight.dtype
                    or not bool(torch.isfinite(controller).all())):
                raise IntegrityError("numeric continuation geometry differs")
            if self.full:
                self.model.validate_state(saved["state"], torch.zeros(1, 1, dtype=torch.long), finite=True)
            elif set(saved["state"]) != {"record_controller"}:
                raise IntegrityError("unexpected numeric-only state")
            if saved["pending_read"] is not None:
                packet_from_dict(saved["pending_read"])
            self.state, self.operations = saved["state"], saved["operations"]
            self.pending_read = saved["pending_read"]
        self.revision = snapshot["revision"]

    def _snapshot_bytes(self, state, operations, pending_read=...):
        pending_read = self.pending_read if pending_read is ... else pending_read
        return _save({"schema": COMPOSITION_SCHEMA, "model_id": self.model_id,
                      "state": state, "operations": operations, "pending_read": pending_read})

    def _vector(self, value) -> Tensor:
        if isinstance(value, dict):
            if value.get("kind") not in {"numeric_vector", "derived_numeric"}:
                raise IntegrityError("record is not a typed numeric vector")
            if value.get("dimension") != self.computer.value_dim:
                raise IntegrityError("numeric record dimension differs")
            value = value.get("values")
        if not isinstance(value, list) or len(value) != self.computer.value_dim:
            raise IntegrityError("numeric record dimension differs")
        if any(type(v) not in (int, float) for v in value):
            raise IntegrityError("numeric record contains a non-number")
        tensor = torch.tensor([value], dtype=self.computer.write_head.weight.dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise IntegrityError("numeric record is not finite")
        return tensor

    def put(self, key: Key, vector, request_id: str, *, source: str | None = None):
        tensor = self._vector(vector)
        value = {"kind": "numeric_vector", "values": tensor[0].tolist(), "dimension": self.computer.value_dim}
        payload = {"method": "put_numeric", "key": list(key), "value": value, "source": source or request_id}
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            self._restore(self.store.refresh_snapshot()); return replay
        snapshot = self.store.snapshot_state(); self._restore(snapshot)
        receipt = self.store.commit_request(request_id, payload,
            operations=[Operation("set", key, value, source or request_id)],
            neural_state=self._snapshot_bytes(self.state, self.operations), reply={"status": "stored"},
            expected_revision=snapshot["revision"])
        self.revision = receipt["revision"]
        return receipt

    def _check_envelope(self, envelope: dict, *, depth: int = 0, seen=()):
        if depth > 16:
            raise StaleDerivedError("derived dependency nesting exceeds the explicit bound")
        if (not isinstance(envelope, dict) or envelope.get("kind") != "derived_numeric"
                or envelope.get("schema") != COMPOSITION_SCHEMA
                or envelope.get("model_id") != self.model_id
                or envelope.get("source_sha256") != source_hash()
                or len(envelope.get("parents", [])) != 2):
            raise IntegrityError("derived result interpretation differs")
        for raw in envelope["parents"]:
            packet = packet_from_dict(raw)
            if packet.key in seen:
                raise StaleDerivedError("cyclic derived dependency")
            if not self.store.validate_packet(packet):
                raise StaleDerivedError("a derived result's supporting record changed")
            if isinstance(packet.value, dict) and packet.value.get("kind") == "derived_numeric":
                self._check_envelope(packet.value, depth=depth + 1, seen=(*seen, packet.key))
        return self._vector(envelope)

    @torch.inference_mode()
    def read_derived(self, key: Key) -> list[float]:
        packet = self.store.resolve(key)
        if packet.status != "ok":
            raise StaleDerivedError("derived record is unavailable")
        value = self._check_envelope(packet.value, seen=(packet.key,))
        # Validation calls have separate read transactions. Strict revision
        # comparison makes their combined result coherent in one history.
        self.store.snapshot_state(expected_revision=packet.revision)
        return value[0].tolist()

    def _consume(self, value: Tensor, state: dict):
        if self.full:
            return self.model.consume_selected(value, state)
        return {"record_controller": self.computer.consume(value, state["record_controller"])}

    def _propose(self, state: dict):
        return (self.model.propose_numeric_write(state) if self.full else
                self.computer.propose(state["record_controller"]))

    @torch.inference_mode()
    def begin_read(self, first: Key, request_id: str):
        """Persist the actual first read and c so a restart can precede read two."""
        if self.model.model_identity() != self.model_id:
            raise IntegrityError("computation model changed")
        payload = {"method": "begin_numeric_read", "first": list(first)}
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            self._restore(self.store.refresh_snapshot()); return replay
        snapshot = self.store.snapshot_state(); self._restore(snapshot)
        if self.pending_read is not None:
            raise ConflictError("finish or explicitly reset the pending numeric read first")
        packet = self.store.resolve(first)
        if packet.revision != snapshot["revision"] or packet.status != "ok":
            raise ConflictError("first numeric support is unavailable or changed")
        value = (self._check_envelope(packet.value, seen=(packet.key,))
                 if isinstance(packet.value, dict) and packet.value.get("kind") == "derived_numeric"
                 else self._vector(packet.value))
        initial = self.model.clear_controller(self.state) if self.full else self._initial()
        state = self._consume(value, initial)
        if self.model.model_identity() != self.model_id:
            raise IntegrityError("computation model changed during the first read")
        receipt = self.store.commit_request(request_id, payload, packet=packet,
            neural_state=self._snapshot_bytes(state, self.operations, packet.to_dict()),
            reply={"status": "first_read_saved", "source": list(first)}, expected_revision=snapshot["revision"])
        self.state, self.pending_read, self.revision = state, packet.to_dict(), receipt["revision"]
        return receipt

    @torch.inference_mode()
    def reset_controller(self, request_id: str):
        """Explicitly cancel a staged computation without modifying factual data."""
        payload = {"method": "reset_numeric_controller"}
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            self._restore(self.store.refresh_snapshot()); return replay
        snapshot = self.store.snapshot_state(); self._restore(snapshot)
        state = self.model.clear_controller(self.state) if self.full else self._initial()
        receipt = self.store.commit_request(request_id, payload,
            neural_state=self._snapshot_bytes(state, self.operations, None),
            reply={"status": "controller_reset"}, expected_revision=snapshot["revision"])
        self.state, self.pending_read, self.revision = state, None, receipt["revision"]
        return receipt

    @torch.inference_mode()
    def finish_read(self, second: Key, destination: Key, request_id: str, *, failpoint=None):
        payload = {"method": "finish_numeric_read", "second": list(second), "destination": list(destination)}
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            self._restore(self.store.refresh_snapshot()); return replay
        snapshot = self.store.snapshot_state(); self._restore(snapshot)
        if self.pending_read is None:
            raise ConflictError("no saved first numeric read")
        first_packet = packet_from_dict(self.pending_read)
        if tuple(destination) in {first_packet.key, tuple(second)}:
            raise ValueError("derived destination must not replace a support")
        if not self.store.validate_packet(first_packet):
            raise StaleDerivedError("saved first-read dependency changed before resumption")
        p1, p2 = self.store.resolve(first_packet.key), self.store.resolve(second)
        if (p1.revision != snapshot["revision"] or p2.revision != snapshot["revision"]
                or p1.status != "ok" or p2.status != "ok"):
            raise ConflictError("resumed supports did not share one current revision")
        for packet in (p1, p2):
            if isinstance(packet.value, dict) and packet.value.get("kind") == "derived_numeric":
                self._check_envelope(packet.value, seen=(tuple(destination), packet.key))
        state = self._consume(self._vector(p2.value), self.state)
        return self._publish_computed(p1, p2, destination, request_id, payload, state,
                                      self._propose(state), snapshot["revision"], False, failpoint)

    @torch.inference_mode()
    def compute(self, first: Key, second: Key, destination: Key, request_id: str, *,
                reset_between: bool = False, failpoint=None):
        if tuple(destination) in {tuple(first), tuple(second)}:
            raise ValueError("derived destination must not replace either supporting record")
        if self.model.model_identity() != self.model_id:
            raise IntegrityError("computation model changed since the session opened")
        payload = {"method": "numeric_two_read", "first": list(first), "second": list(second),
                   "destination": list(destination), "reset_between": bool(reset_between)}
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            self._restore(self.store.refresh_snapshot()); return replay
        snapshot = self.store.snapshot_state(); self._restore(snapshot)
        if self.pending_read is not None:
            raise ConflictError("finish or explicitly reset the pending numeric read first")
        revision = snapshot["revision"]
        p1, p2 = self.store.resolve(first), self.store.resolve(second)
        if p1.revision != revision or p2.revision != revision:
            raise ConflictError("support reads did not share the continuation's revision")
        if p1.status != "ok" or p2.status != "ok":
            raise ConflictError("numeric support is missing, retracted or pending")
        values = []
        for packet in (p1, p2):
            value = packet.value
            if isinstance(value, dict) and value.get("kind") == "derived_numeric":
                values.append(self._check_envelope(value, seen=(tuple(destination), packet.key)))
            else:
                values.append(self._vector(value))
        state = self.model.clear_controller(self.state) if self.full else self._initial()
        if self.full:
            state = self.model.consume_selected(values[0], state)
            if reset_between:
                state = self.model.clear_controller(state)
            state = self.model.consume_selected(values[1], state)
            proposed = self.model.propose_numeric_write(state)
        else:
            c = self.computer.consume(values[0], state["record_controller"])
            if reset_between:
                c = torch.zeros_like(c)
            c = self.computer.consume(values[1], c)
            state = {"record_controller": c}
            proposed = self.computer.propose(c)
        return self._publish_computed(p1, p2, destination, request_id, payload, state,
                                      proposed, revision, reset_between, failpoint)

    def _publish_computed(self, p1, p2, destination, request_id, payload, state,
                          proposed, revision, reset_between, failpoint):
        if not bool(torch.isfinite(proposed).all()):
            raise IntegrityError("learned numeric writer proposed a nonfinite value")
        envelope = {"kind": "derived_numeric", "schema": COMPOSITION_SCHEMA,
                    "dimension": self.computer.value_dim, "values": proposed[0].tolist(),
                    "model_id": self.model_id, "source_sha256": source_hash(),
                    "parents": [p1.to_dict(), p2.to_dict()], "reset_between": bool(reset_between),
                    "interface": "addresses-and-destination-supplied"}
        # Both dependencies are checked, then the immutable common revision is
        # compared again inside the atomic publication transaction. Passing p1
        # also exercises Store's normal post-operation packet validation.
        if not self.store.validate_packet(p1) or not self.store.validate_packet(p2):
            raise ConflictError("numeric support changed before publication")
        if self.model.model_identity() != self.model_id:
            raise IntegrityError("computation model changed before publication")
        try:
            receipt = self.store.commit_request(request_id, payload,
                operations=[Operation("set", destination, envelope, request_id)], packet=p1,
                neural_state=self._snapshot_bytes(state, self.operations + 1, None),
                reply={"status": "derived", "values": proposed[0].tolist(),
                       "destination": list(destination), "parents": [p1.packet_digest, p2.packet_digest]},
                expected_revision=revision, failpoint=failpoint)
        except BaseException:
            self._restore(self.store.refresh_snapshot()); raise
        if receipt["replayed"]:
            self._restore(self.store.refresh_snapshot())
        else:
            self.state, self.operations, self.revision = state, self.operations + 1, receipt["revision"]
            self.pending_read = None
        return receipt

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
