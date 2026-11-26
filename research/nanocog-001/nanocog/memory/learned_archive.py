"""Experimental categorical factual mode over a frozen Connected COG model.

This is a bounded learned writer/query/answer adapter, not general language
memory. The native language model remains intact. Record typing, atomic storage,
and at most two-hop traversal are exact host infrastructure. No gold labels,
route overrides, or text-to-operation parser are used by this inference path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .store import ConflictError, IntegrityError, MemoryStore


SCHEMA = "cog-factual-adapter-v1"
OWNERS = ("Amara", "Bela", "Cato", "Dina", "Evan", "Fara", "Gus", "Hana")
BOXES = ("chest", "crate", "basket", "trunk")
ITEMS = ("coin", "ring", "watch", "key")
ANSWERS = OWNERS + ("unknown",)
UNKNOWN = 12
N_OPERATIONS = 97


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fact_key(record: int) -> str:
    if type(record) is not int or not 0 <= record < 8:
        raise ValueError("record must be an integer from zero to seven")
    return "cog-factual:" + (("holder:" + BOXES[record]) if record < 4
                             else ("container:" + ITEMS[record - 4]))


def operation(record: int, value: int) -> int:
    if type(record) is not int or type(value) is not int or not (0 <= record < 8 and 0 <= value < 12):
        raise ValueError("operation categories outside the schema")
    return 1 + 12 * record + value


def unpack_operation(index: int) -> tuple[int, int] | None:
    if type(index) is not int or not 0 <= index < N_OPERATIONS:
        raise ValueError("operation outside the schema")
    return None if index == 0 else divmod(index - 1, 12)


def valid_write(record: int, value: int) -> bool:
    return (0 <= record < 4 and 0 <= value < 8) or (4 <= record < 8 and 8 <= value < 12)


def fact_payload(record: int, value: int) -> dict:
    if not valid_write(record, value):
        raise ValueError("record/value types do not match")
    return {"schema": SCHEMA, "kind": "owner" if record < 4 else "box",
            "value": OWNERS[value] if record < 4 else BOXES[value - 8]}


def payload_value(record: int, payload: Any) -> int:
    if not isinstance(payload, dict) or set(payload) != {"schema", "kind", "value"}:
        raise IntegrityError("accepted categorical fact has an invalid shape")
    if payload["schema"] != SCHEMA or payload["kind"] != ("owner" if record < 4 else "box"):
        raise IntegrityError("accepted categorical fact has an incompatible schema/type")
    labels = OWNERS if record < 4 else BOXES
    if payload["value"] not in labels:
        raise IntegrityError("accepted categorical fact is outside the ontology")
    return labels.index(payload["value"]) + (0 if record < 4 else 8)


def execute_operation(ledger: list[int], index: int) -> str:
    """Exact diagnostic executor. Invalid proposals are rejected, never repaired."""
    if len(ledger) != 8:
        raise ValueError("ledger requires eight records")
    proposal = unpack_operation(index)
    if proposal is None:
        return "noop"
    record, value = proposal
    if not valid_write(record, value):
        return "rejected_invalid"
    ledger[record] = value
    return "written"


def read_ledger(ledger: Sequence[int], route: int) -> int:
    """Exact typed traversal, exposed separately from the learned answer head."""
    if len(ledger) != 8 or type(route) is not int or not 0 <= route < 8:
        raise ValueError("invalid categorical ledger or route")
    value = int(ledger[route])
    if route >= 4:
        return int(ledger[value - 8]) if 8 <= value < 12 else UNKNOWN
    return value if 0 <= value < 8 else UNKNOWN


class FactualAdapter(nn.Module):
    """Frozen full native LM plus 18,567 learned factual parameters.

    ``native`` must expose the actual ConnectedCOGTinyLM methods. Admission and
    source/checkpoint hashes are the loader's responsibility. Feature extraction
    runs the six native blocks AND the connected record board with no overrides.
    Each bounded event/query starts with fresh native transient state; history
    lives in the durable accepted-fact archive, not a replayed prompt.
    """

    def __init__(self, native: nn.Module, chars: Sequence[str], *,
                 admission: dict[str, Any] | None = None) -> None:
        super().__init__()
        if getattr(native.config, "architecture_id", None) != "cog-unified-connected-v2":
            raise ValueError("the adapter requires Connected COG v2")
        if native.config.d_model != 122 or len(chars) != native.config.vocab_size or len(set(chars)) != len(chars):
            raise ValueError("incompatible native feature width or tokenizer")
        self.native = native
        self.chars = tuple(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.admission = json.loads(_json(admission or {"status": "caller-supplied-unverified"}))
        if any(not isinstance(c, str) or len(c) != 1 for c in chars):
            raise ValueError("tokenizer entries must be individual characters")
        self.native.requires_grad_(False)
        self.native.eval()
        self.encoder = nn.GRU(122, 32, batch_first=True)
        self.writer = nn.Linear(32, N_OPERATIONS)
        self.query = nn.Linear(32, 8)
        self.answer = nn.Linear(13, 9)

    def train(self, mode: bool = True):
        super().train(mode)
        self.native.eval()
        return self

    @torch.no_grad()
    def native_features(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        if not texts or any(not isinstance(t, str) or not t or len(t) > 256 for t in texts):
            raise ValueError("events/queries must contain one to 256 characters")
        unknown = set("".join(texts)) - self.stoi.keys()
        if unknown:
            raise ValueError("text contains characters absent from the admitted tokenizer")
        device = next(self.native.parameters()).device
        lengths = torch.tensor([len(t) for t in texts], dtype=torch.long, device=device)
        ids = torch.full((len(texts), int(lengths.max())), self.stoi.get(" ", 0),
                         dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            ids[i, :len(text)] = torch.tensor([self.stoi[c] for c in text], device=device)
        state = self.native.init_state(len(texts), device)
        features, _ = self.native.backbone_features(ids, state)
        features, _, _, _ = self.native.record_board(
            features, state["memory"], state["controller"],
            feedback_enabled=True, memory_enabled=True, return_trace=False, overrides=None)
        return self.native.final_norm(features).detach(), lengths

    def encode_features(self, features: Tensor, lengths: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != 122 or lengths.shape != (len(features),):
            raise ValueError("invalid native features/lengths")
        # Padding is only after the final real character, so gathering before it
        # is equivalent to a packed causal GRU and uses the faster dense kernel.
        outputs, _ = self.encoder(features)
        return outputs[torch.arange(len(features), device=features.device), lengths.long() - 1]

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encode_features(features, lengths)
        return self.writer(encoded), self.query(encoded)

    @torch.inference_mode()
    def predict_events(self, texts: Sequence[str]) -> list[int]:
        features, lengths = self.native_features(texts)
        return self(features, lengths)[0].argmax(-1).tolist()

    @torch.inference_mode()
    def predict_queries(self, texts: Sequence[str]) -> list[int]:
        features, lengths = self.native_features(texts)
        return self(features, lengths)[1].argmax(-1).tolist()

    def answer_logits(self, values: Tensor) -> Tensor:
        return self.answer(F.one_hot(values.long(), 13).to(self.answer.weight.dtype))

    @torch.inference_mode()
    def predict_answers(self, values: Sequence[int]) -> list[int]:
        tensor = torch.tensor(values, device=self.answer.weight.device)
        return self.answer_logits(tensor).argmax(-1).tolist()

    def model_identity(self) -> str:
        h = hashlib.sha256(_json([SCHEMA, self.chars, self.stoi, OWNERS, BOXES, ITEMS,
                                  self.admission, asdict(self.native.config),
                                  self.native.default_feedback_enabled,
                                  self.native.record_board.use_scripted_core,
                                  "fresh-state;feedback=true;memory=true;no-overrides"]).encode())
        for name, value in sorted(self.state_dict().items()):
            h.update(name.encode())
            h.update(str((tuple(value.shape), value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class WriteResult:
    operation: int
    status: str
    decision_event_id: str
    fact: dict | None
    replayed: bool


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    answer_index: int
    route: int
    read_value: int
    dependencies: dict[str, int]
    assistant_event_id: str
    replayed: bool


class FactualMemory:
    """Automatic durable observations plus a learned, bounded factual interface.

    The explicit ``authority`` is a host acceptance policy, not a truth oracle.
    An accepted prediction can be false. Successful replies are returned only
    after their actual read dependencies have been revalidated in the commit.
    """

    def __init__(self, store: MemoryStore, adapter: FactualAdapter, *,
                 session_id: str, authority: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256:
            raise ValueError("invalid session_id")
        if not isinstance(authority, str) or not authority.strip() or len(authority) > 256:
            raise ValueError("an explicit host authority is required")
        self.store, self.adapter = store, adapter.eval()
        self.session_id, self.authority = session_id, authority
        self.model_id = adapter.model_identity()

    def _check_model(self) -> None:
        # Identity checks deliberately include all native/adapter tensor bytes.
        # This cost is included in serving measurements. Host code/class changes
        # remain within the admitted loader's trust boundary, not a sandbox.
        global_hooks = torch.nn.modules.module
        if (self.adapter.training or any(m.training for m in self.adapter.native.modules()) or
                global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks or
                any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                    for m in self.adapter.modules()) or
                self.adapter.model_identity() != self.model_id):
            raise IntegrityError("factual model changed after attachment")

    def _ids(self, request_id: str, kind: str) -> tuple[str, str, str]:
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 256:
            raise ValueError("invalid request_id")
        h = hashlib.sha256(_json([SCHEMA, self.session_id, request_id, kind]).encode()).hexdigest()
        return "factual:" + h + ":source", "factual:" + h + ":result", "factual:" + h + ":proposal"

    def _receipt(self, row: dict, kind: str) -> dict:
        try:
            content = json.loads(row["content"])
            if (row["session_id"] != self.session_id or content["schema"] != SCHEMA or
                    content["model_id"] != self.model_id or content["kind"] != kind):
                raise ValueError("incompatible receipt")
        except (ValueError, TypeError, KeyError) as exc:
            raise IntegrityError("stored factual receipt has an incompatible identity") from exc
        return content

    def observe(self, text: str, *, request_id: str) -> WriteResult:
        self._check_model()
        source_id, decision_id, proposal_id = self._ids(request_id, "observe")
        self.store.append_message(self.session_id, "user", text, event_id=source_id)
        previous = self.store.get_message(decision_id)
        replayed = previous is not None
        if previous is None:
            # The route is not known until inference. Freeze the bounded set of
            # candidate revisions first, so a concurrently completed write is
            # never silently adopted as the base of an older in-flight event.
            # This guards changes after each snapshot, not a total event-time
            # ordering among concurrent conversations.
            base_revisions = {r: self.store.fact_snapshot(fact_key(r))["revision"] for r in range(8)}
            index = int(self.adapter.predict_events([text])[0])
            self._check_model()
            proposed = unpack_operation(index)
            status, revision = "noop", None
            if proposed is not None:
                record, value = proposed
                status = "written" if valid_write(record, value) else "rejected_invalid"
                if status == "written":
                    revision = base_revisions[record]
            receipt = {"schema": SCHEMA, "kind": "decision", "model_id": self.model_id,
                       "operation": index, "status": status, "base_revision": revision,
                       "authority": self.authority}
            previous = self.store.append_message(self.session_id, "tool", _json(receipt),
                                                  event_id=decision_id, source_event_ids=[source_id])
        receipt = self._receipt(previous, "decision")
        if previous["role"] != "tool" or receipt.get("authority") != self.authority:
            raise ConflictError("stored decision authority cannot be changed")
        index, status = receipt["operation"], receipt["status"]
        proposed = unpack_operation(index)
        expected_status = "noop" if proposed is None else ("written" if valid_write(*proposed) else "rejected_invalid")
        if status != expected_status:
            raise IntegrityError("stored decision has inconsistent operation/status")
        fact = None
        if status == "written":
            record, value = proposed
            # Persist the captured revision in the immutable decision first. A
            # retry completes this same proposal; it never rebases old text.
            self.store.propose(fact_key(record), fact_payload(record, value), decision_id,
                               proposal_id=proposal_id, expected_revision=receipt["base_revision"])
            fact = self.store.accept(proposal_id, authority=self.authority)
        return WriteResult(index, status, decision_id, fact, replayed)

    def ask(self, text: str, *, request_id: str) -> AnswerResult:
        self._check_model()
        source_id, response_id, _ = self._ids(request_id, "ask")
        self.store.append_message(self.session_id, "user", text, event_id=source_id)
        previous = self.store.get_message(response_id)
        if previous is not None:
            receipt = self._receipt(previous, "answer")
            if previous["role"] != "assistant":
                raise IntegrityError("stored response has the wrong role")
            return self._answer_result(receipt, response_id, True)
        route = int(self.adapter.predict_queries([text])[0])
        if not 0 <= route < 8:
            raise ValueError("model returned an invalid query route")
        revisions: dict[str, int] = {}
        sources = {source_id}

        def read(record: int) -> int:
            key = fact_key(record)
            snapshot = self.store.fact_snapshot(key)
            revisions[key] = snapshot["revision"]
            fact = snapshot["record"]
            if fact is None:
                return UNKNOWN
            sources.add(fact["source_event_id"])
            return payload_value(record, fact["value"])

        value = read(route)
        if route >= 4:
            value = read(value - 8) if 8 <= value < 12 else UNKNOWN
        answer = int(self.adapter.predict_answers([value])[0])
        self._check_model()
        if not 0 <= answer < len(ANSWERS):
            raise ValueError("model returned an invalid answer category")
        receipt = {"schema": SCHEMA, "kind": "answer", "model_id": self.model_id,
                   "route": route, "read_value": value, "answer_index": answer,
                   "answer": ANSWERS[answer], "dependencies": revisions}
        saved = self.store.append_message(self.session_id, "assistant", _json(receipt),
                                          event_id=response_id, source_event_ids=sorted(sources),
                                          expected_fact_revisions=revisions)
        return self._answer_result(self._receipt(saved, "answer"), response_id, False)

    @staticmethod
    def _answer_result(receipt: dict, event_id: str, replayed: bool) -> AnswerResult:
        return AnswerResult(receipt["answer"], receipt["answer_index"], receipt["route"],
                            receipt["read_value"], receipt["dependencies"], event_id, replayed)
