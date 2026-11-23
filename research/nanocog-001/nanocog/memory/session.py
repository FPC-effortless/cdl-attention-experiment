"""Automatic conversation capture and bounded, source-labelled recall.

The host chooses the principal and namespace when it creates ``MemoryStore``.
This module does not authenticate users, call a language model, or turn retrieved
text into instructions. A host supplies its own response callback. Source text
is remembered automatically; interpreting it as an accepted fact is a separate
operation on the store.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable
import uuid

from .store import MemoryStore


def context_json(context: dict[str, Any]) -> str:
    """Canonical rendering used for the context-size budget (Unicode characters)."""
    return json.dumps(context, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class PreparedTurn:
    request_id: str
    session_id: str
    user_event_id: str
    assistant_event_id: str
    context: dict[str, Any]


@dataclass(frozen=True)
class TurnResult:
    response: str
    request_id: str
    session_id: str
    user_event_id: str
    assistant_event_id: str
    replayed: bool


class MemorySession:
    """A chat lifecycle whose successful messages are saved without ``save()``.

    Reuse the same store principal/namespace in another chat to recall its
    observations and accepted facts. Use a new session ID for a new conversation.
    To retry a turn after losing its acknowledgement, reuse its request ID.

    ``respond(message, context)`` should be side-effect-free. A failed callback
    leaves the already committed user observation available for retry. The
    callback itself cannot be made exactly-once across process failures: a crash
    between computation and recording the response may cause recomputation.
    A committed response is returned directly on an exact retry.
    """

    def __init__(self, store: MemoryStore, *, session_id: str | None = None,
                 recall_limit: int = 8, max_context_chars: int = 8000) -> None:
        if type(recall_limit) is not int or not 1 <= recall_limit <= 100:
            raise ValueError("recall_limit must be an integer from 1 to 100")
        if type(max_context_chars) is not int or not 256 <= max_context_chars <= 1_000_000:
            raise ValueError("max_context_chars must be an integer from 256 to 1000000")
        chosen = session_id if session_id is not None else str(uuid.uuid4())
        if not isinstance(chosen, str) or not chosen.strip() or len(chosen) > 256:
            raise ValueError("session_id must be a nonempty string of at most 256 characters")
        self.store = store
        self.session_id = chosen
        self.recall_limit = recall_limit
        self.max_context_chars = max_context_chars

    def context(self, query: str, *, exclude_event_id: str | None = None) -> dict[str, Any]:
        """Return bounded data. A caller must keep it below its instruction layer.

        Whole records are selected, never silently shortened into a different
        claim. The budget includes JSON field names, provenance and punctuation.
        This is a character budget, not a model-specific tokenizer budget.
        """
        raw = self.store.recall(query, limit=self.recall_limit,
                                max_chars=self.max_context_chars,
                                exclude_event_id=exclude_event_id)
        result: dict[str, Any] = {
            "schema_version": 1,
            "accepted_facts": [],
            "unverified_observations": [],
            "truncated": bool(raw.get("truncated", False)),
        }
        selected = 0
        for category, records in (("accepted_facts", raw.get("facts", [])),
                                  ("unverified_observations", raw.get("observations", []))):
            for original in records:
                if category == "unverified_observations" and original.get("event_id") == exclude_event_id:
                    continue
                if selected >= self.recall_limit:
                    result["truncated"] = True
                    continue
                # Copy: callers and callbacks cannot mutate store-owned records.
                item = json.loads(context_json(original))
                if category == "unverified_observations":
                    item["epistemic_status"] = "unverified_observation"
                    item["source_event_id"] = item["event_id"]
                else:
                    item["epistemic_status"] = "accepted_by_host"
                result[category].append(item)
                if len(context_json(result)) > self.max_context_chars:
                    result[category].pop()
                    result["truncated"] = True
                else:
                    selected += 1
        return result

    def _record_user(self, user_message: str, request_id: str | None) -> tuple[str, str, str]:
        request_id = str(uuid.uuid4()) if request_id is None else request_id
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 256:
            raise ValueError("request_id must be a nonempty string of at most 256 characters")
        identity = json.dumps([self.session_id, request_id], ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        user_event_id, assistant_event_id = f"turn:{digest}:user", f"turn:{digest}:assistant"
        self.store.append_message(self.session_id, "user", user_message,
                                  event_id=user_event_id)
        return request_id, user_event_id, assistant_event_id

    def prepare(self, user_message: str, *, request_id: str | None = None) -> PreparedTurn:
        """Persist a user message, then gather prior memory for a response."""
        request_id, user_event_id, assistant_event_id = self._record_user(user_message, request_id)
        return PreparedTurn(request_id, self.session_id, user_event_id,
                            assistant_event_id,
                            self.context(user_message, exclude_event_id=user_event_id))

    def turn(self, user_message: str,
             respond: Callable[[str, dict[str, Any]], str], *,
             request_id: str | None = None) -> TurnResult:
        """Capture input, recall context, compute, and durably capture the reply."""
        if not callable(respond):
            raise TypeError("respond must be callable")
        request_id, user_event_id, assistant_event_id = self._record_user(user_message, request_id)
        previous = self.store.get_message(assistant_event_id)
        if previous is not None:
            if previous["session_id"] != self.session_id or previous["role"] != "assistant":
                raise ValueError("stored response has the wrong conversation identity")
            return TurnResult(previous["content"], request_id, self.session_id,
                              user_event_id, assistant_event_id, True)
        context = self.context(user_message, exclude_event_id=user_event_id)
        # Capture provenance before the callback can mutate its context object.
        # The store rechecks that every source is still live inside the commit.
        sources = {user_event_id}
        for category in ("accepted_facts", "unverified_observations"):
            for item in context[category]:
                source = item.get("source_event_id")
                if source is not None:
                    sources.add(source)
        expected_facts = {item["key"]: item["revision"] for item in context["accepted_facts"]}
        response = respond(user_message, context)
        if not isinstance(response, str):
            raise TypeError("response callback must return a string")
        saved = self.store.append_message(self.session_id, "assistant", response,
                                          event_id=assistant_event_id,
                                          source_event_ids=sorted(sources),
                                          expected_fact_revisions=expected_facts)
        return TurnResult(saved["content"], request_id, self.session_id,
                          user_event_id, assistant_event_id, False)
