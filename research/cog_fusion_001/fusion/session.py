"""Durable learned-text bridge for the fused COG experiment.

Observations and questions each traverse the continuous neural backbone once.
The learned interpreter predicts operations/query identities. Exact lookup
supplies a bounded evidence packet to the real character decoder. SQLite binds
the new continuation, private RNG, factual changes and reply in one commit.

Pending abstention is an explicit runtime policy, not a learned capability.
Semantic correctness is evaluated elsewhere; successful publication alone does
not establish that an operation or answer was the right one for the language.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import wraps
import io
import math
from pathlib import Path
import threading
from typing import Any, Iterable

import torch

from .data import ENTITIES, RELATIONS, UNKNOWN
from .engine import Resolution, evidence_tensors, greedy_generate, predicted_operation
from .model import FusionModel
from .store import ConflictError, EvidencePacket, IntegrityError, Operation, Store


SESSION_SCHEMA = "cog-fusion-session-v1"
PREFIX = "\nAnswer: "
UNKNOWN_VALUE = "unknown"


def _serialized(method):
    """A live Session is single-flight, including same-thread reentrant hooks."""
    @wraps(method)
    def run(self, *args, **kwargs):
        if not self._request_lock.acquire(blocking=False):
            raise ConflictError("this session is already processing a request")
        try:
            return method(self, *args, **kwargs)
        finally:
            self._request_lock.release()
    return run


def _serialize(value: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _deserialize(value: bytes) -> dict:
    try:
        result = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise IntegrityError("continuation cannot be decoded as tensor-only state") from exc
    if not isinstance(result, dict):
        raise IntegrityError("continuation must be a dictionary")
    return result


class Session:
    """One model, one scope, one continuous stream, and an isolated sampling RNG."""

    def __init__(self, model: FusionModel, path: str | Path, *, scope: str = "world",
                 capacity: int | None = None, seed: int = 0, max_chars: int = 12):
        if not isinstance(model, FusionModel):
            raise TypeError("model must be FusionModel")
        if model.native.embedding.weight.device.type != "cpu":
            raise ValueError("this qualified durable runtime is CPU-only")
        if not isinstance(scope, str) or not scope or len(scope) > 512:
            raise ValueError("scope must be a nonempty identifier")
        if type(max_chars) is not int or not 1 <= max_chars <= 12:
            raise ValueError("max_chars must be in 1..12")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.model = model.eval()
        self.model_id = model.model_identity()
        self.scope = scope
        self.seed = seed
        self.max_chars = max_chars
        self._request_lock = threading.Lock()
        self._model_contract = self._contract()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.store = Store(path, self.model_id, schema_id=SESSION_SCHEMA, capacity=capacity)
        self.state = model.init_state(1)
        self.counters = self._new_counters()
        self.revision = 0
        try:
            self._restore(self.store.snapshot_state())
        except BaseException:
            self.store.close()
            raise

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, path: str | Path, **kwargs) -> "Session":
        return cls(FusionModel.from_checkpoint(checkpoint), path, **kwargs)

    def _contract(self):
        return (tuple((name, id(value), value._version) for name, value in self.model.named_parameters()),
                asdict(self.model.config), asdict(self.model.native.config),
                self.model.memory_enabled, self.model.chars,
                tuple((name, module.training) for name, module in self.model.named_modules()))

    def _check_model(self):
        if self._contract() != self._model_contract or self.model.training:
            raise IntegrityError("model weights/configuration/training mode changed during the session")

    @staticmethod
    def _new_counters():
        return {"input_tokens": 0, "emitted_tokens": 0, "control_tokens": 0,
                "requests": 0, "observations": 0, "questions": 0,
                "record_updates": 0, "historical_tokens_replayed": 0}

    def _restore(self, snapshot: dict):
        if snapshot["neural_state"]:
            saved = _deserialize(snapshot["neural_state"])
            expected = {"schema", "model_id", "scope", "max_chars", "state", "counters"}
            if (set(saved) != expected or saved["schema"] != SESSION_SCHEMA
                    or saved["model_id"] != self.model_id or saved["scope"] != self.scope
                    or saved["max_chars"] != self.max_chars):
                raise IntegrityError("continuation interpretation does not match this session")
            counters = saved["counters"]
            if (not isinstance(counters, dict) or set(counters) != set(self._new_counters())
                    or any(type(value) is not int or value < 0 for value in counters.values())
                    or counters["historical_tokens_replayed"] != 0):
                raise IntegrityError("invalid continuation token accounting")
            try:
                self.model.validate_state(saved["state"], torch.zeros((1, 1), dtype=torch.long), finite=True)
            except Exception as exc:
                raise IntegrityError("saved recurrent state has incompatible geometry") from exc
            random = _deserialize(snapshot["rng_state"])
            if set(random) != {"torch_generator"}:
                raise IntegrityError("invalid sampling state")
            try:
                self.generator.set_state(random["torch_generator"])
            except Exception as exc:
                raise IntegrityError("invalid sampling state tensor") from exc
            self.state, self.counters = saved["state"], dict(counters)
        else:
            if snapshot["rng_state"]:
                raise IntegrityError("RNG exists without a matching neural continuation")
            self.state, self.counters = self.model.init_state(1), self._new_counters()
            self.generator.manual_seed(self.seed)
        self.revision = snapshot["revision"]

    @_serialized
    def refresh(self) -> None:
        """Explicitly adopt another connection's coherent authoritative state."""
        self._check_model()
        self._restore(self.store.refresh_snapshot())

    def _begin(self, request_id: str, payload: dict) -> dict | None:
        self._check_model()
        replay = self.store.lookup_receipt(request_id, payload)
        if replay is not None:
            # The receipt may be historical, but the continuation must remain
            # the latest authoritative one; replay never feeds old text again.
            self._restore(self.store.refresh_snapshot())
            return self._result(replay)
        snapshot = self.store.snapshot_state()
        if snapshot["revision"] != self.revision:
            self._restore(snapshot)
        return None

    def _state_blob(self, state: dict, counters: dict) -> bytes:
        return _serialize({"schema": SESSION_SCHEMA, "model_id": self.model_id,
                           "scope": self.scope, "max_chars": self.max_chars,
                           "state": state, "counters": counters})

    def _rng_blob(self) -> bytes:
        return _serialize({"torch_generator": self.generator.get_state()})

    @staticmethod
    def _result(receipt: dict) -> dict:
        result = dict(receipt["reply"])
        result.update(request_id=receipt["request_id"], revision=receipt["revision"],
                      branch_id=receipt["branch_id"], replayed=receipt["replayed"],
                      historical_receipt=receipt["historical_receipt"])
        return result

    def _publish(self, request_id: str, payload: dict, state: dict, counters: dict,
                 reply: dict, *, expected_revision: int, operations=(), packet=None,
                 clear_pending=(), failpoint=None):
        try:
            # The hosting process owns the model. Ordinary mutation is detected
            # both before and after inference. Deliberate .data manipulation or
            # arbitrary monkeypatching is outside this trusted-host contract.
            self._check_model()
            receipt = self.store.commit_request(
                request_id, payload, operations=operations,
                neural_state=self._state_blob(state, counters), rng_state=self._rng_blob(),
                reply=reply, packet=packet, clear_pending=clear_pending,
                expected_revision=expected_revision, failpoint=failpoint)
        except BaseException:
            # A failure *after* COMMIT means the new state already won. Reload
            # instead of guessing whether to keep the old or new process state.
            self._restore(self.store.refresh_snapshot())
            raise
        if receipt["replayed"]:
            self._restore(self.store.refresh_snapshot())
        else:
            self.state, self.counters, self.revision = state, counters, receipt["revision"]
        return self._result(receipt)

    @staticmethod
    def _text(text: str) -> str:
        if not isinstance(text, str) or not text or len(text) > 65536:
            raise ValueError("text must contain 1..65536 characters")
        return text + "\n"

    @_serialized
    @torch.inference_mode()
    def observe(self, text: str, request_id: str, *, clear_pending: Iterable[str] = (),
                failpoint=None) -> dict:
        event_text = self._text(text)
        clear_pending = tuple(clear_pending)
        payload = {"method": "observe", "scope": self.scope, "text": text,
                   "clear_pending": list(clear_pending)}
        replay = self._begin(request_id, payload)
        if replay is not None:
            return replay
        starting_revision = self.revision
        tokens, lengths = self.model.encode_texts([event_text])
        _, semantics, next_state = self.model.encode_event(tokens, lengths, self.state)
        op, entity, relation, value = predicted_operation(semantics, 0)
        key = (self.scope, ENTITIES[entity], RELATIONS[relation])
        operations = []
        if op == 1:
            name = UNKNOWN_VALUE if value == UNKNOWN else ENTITIES[value]
            operations.append(Operation("link" if relation == 1 else "set", key, name, request_id))
        elif op == 2:
            operations.append(Operation("delete", key, source=request_id))
        elif op != 0:
            raise IntegrityError("model operation is outside the declared executor vocabulary")
        counters = dict(self.counters)
        counters["input_tokens"] += len(event_text)
        counters["requests"] += 1
        counters["observations"] += 1
        counters["record_updates"] += len(operations)
        reply = {"status": "committed", "predicted_operation": [op, entity, relation, value],
                 "record_updated": bool(operations), "key": list(key) if operations else None}
        return self._publish(request_id, payload, next_state, counters, reply,
                             expected_revision=starting_revision, operations=operations,
                             clear_pending=clear_pending, failpoint=failpoint)

    @staticmethod
    def _resolution(packet: EvidencePacket, plan: int) -> Resolution:
        def value_id(value):
            if value == UNKNOWN_VALUE:
                return UNKNOWN
            if not isinstance(value, str) or value not in ENTITIES:
                raise IntegrityError("stored value is outside the model's admitted entity vocabulary")
            return ENTITIES.index(value)
        values = [value_id(value) for value in packet.values]
        if plan == 2:
            values = (values + [UNKNOWN, UNKNOWN])[:2]
        else:
            values = (values + [UNKNOWN])[:1]
        keys = []
        for dep in packet.dependencies:
            _, entity, relation = dep.key
            if entity == UNKNOWN_VALUE:
                continue
            if entity not in ENTITIES or relation not in RELATIONS:
                raise IntegrityError("retrieved identity is outside the model's admitted vocabulary")
            keys.append((ENTITIES.index(entity), RELATIONS.index(relation)))
        return Resolution(values, keys, "ok" if values[-1] != UNKNOWN else "missing",
                          packet.read_count)

    def _sample_generate(self, state: dict, resolution: Resolution, temperature: float):
        tokens, _ = self.model.encode_texts([PREFIX])
        values, masks = evidence_tensors([resolution])
        logits, state = self.model.decode(tokens, state, values, masks)
        generated = ""
        for _ in range(self.max_chars):
            probabilities = (logits[:, -1] / temperature).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise IntegrityError("nonfinite sampling probabilities")
            token = torch.multinomial(probabilities, 1, generator=self.generator)
            char = self.model.chars[int(token[0, 0])]
            generated += char
            logits, state = self.model.decode(token, state, values, masks)
            if char == ".":
                break
        return generated, state

    @_serialized
    @torch.inference_mode()
    def ask(self, text: str, request_id: str, *, temperature: float = 0,
            failpoint=None) -> dict:
        query_text = self._text(text)
        if isinstance(temperature, bool):
            raise ValueError("temperature must be a finite nonnegative number")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        payload = {"method": "ask", "scope": self.scope, "text": text, "temperature": temperature}
        replay = self._begin(request_id, payload)
        if replay is not None:
            return replay
        starting_revision = self.revision
        tokens, lengths = self.model.encode_texts([query_text])
        _, semantics, next_state = self.model.encode_event(tokens, lengths, self.state)
        entity = int(semantics["query_entity"][0].argmax())
        plan = int(semantics["query_plan"][0].argmax())
        key = (self.scope, ENTITIES[entity], RELATIONS[1 if plan in (1, 2) else 0])
        packet = self.store.resolve(key, "owner" if plan == 2 else None)
        counters = dict(self.counters)
        counters["input_tokens"] += len(query_text)
        counters["requests"] += 1
        counters["questions"] += 1
        if packet.status == "pending":
            # This response is a deterministic safety policy, reported separately
            # from both neural answer quality and learned unknown generation.
            answer = "pending"
            reply = {"status": "pending", "text": answer, "generation_mode": "runtime_policy",
                     "pending_ids": list(packet.pending)}
        else:
            resolution = self._resolution(packet, plan)
            original_rng = self.generator.get_state()
            try:
                if temperature == 0:
                    answers, next_state = greedy_generate(self.model, next_state, [resolution],
                                                           max_chars=self.max_chars)
                    answer = answers[0]
                else:
                    answer, next_state = self._sample_generate(next_state, resolution, temperature)
            except BaseException:
                self.generator.set_state(original_rng)
                raise
            counters["control_tokens"] += len(PREFIX)
            counters["emitted_tokens"] += len(answer)
            reply = {"status": "generated", "text": answer, "generation_mode": "neural"}
        reply.update(evidence_status=packet.status, query_entity=ENTITIES[entity], query_plan=plan,
                     selected_values=list(packet.values), records_read=packet.read_count,
                     dependencies=[asdict(dependency) for dependency in packet.dependencies])
        return self._publish(request_id, payload, next_state, counters, reply,
                             expected_revision=starting_revision, packet=packet, failpoint=failpoint)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_args):
        self.close()
