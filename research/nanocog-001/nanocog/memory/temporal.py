"""Explicit action--cue bindings and a finite temporal reference interpreter.

This is an engineered interface/diagnostic, not a language parser or a learned
repair. PHASE-003 found that extra absolute phase labels did not qualify the
event selector; its proposed next experiment binds a cue to an action and its
scope before selecting an event. The interface below makes that relationship
representable and inspectable without claiming to have learned it:
https://github.com/FPC-effortless/cdl-attention-experiment/blob/
6e267541aa2c5095585e5c823fc0eef44ef7d0c3/research/nanocog-001/
cog-bind-time-phase-003/RESULT.md

Action IDs name events in the described world. ``source_event_id`` names the
record that supplies evidence. Source ingestion order, IDs, and recording
timestamps never determine actual event order. All offsets are Python Unicode
character offsets, not UTF-8 byte offsets. Valid spans prove faithful quotation,
not correct linguistic attachment or factual truth. New claims are proposals;
only explicitly accepted claims enter the reference graph.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

BEFORE, AFTER, UNKNOWN = "BEFORE", "AFTER", "UNKNOWN"
_STATUSES = frozenset({"proposed", "accepted", "rejected", "ambiguous"})


def _identifier(value: str, field: str) -> None:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"{field} must be a nonempty explicit identifier")


def _relation(value: str) -> None:
    if value not in (BEFORE, AFTER):
        raise ValueError("relation must be BEFORE or AFTER")


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start
                or not isinstance(self.text, str)
                or len(self.text) != self.end - self.start):
            raise ValueError("span requires nonempty text and exact character offsets")

    def validate(self, source_text: str) -> None:
        if not isinstance(source_text, str):
            raise ValueError("source text must be a string")
        if self.end > len(source_text) or source_text[self.start:self.end] != self.text:
            raise ValueError("span does not match its source text")

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceSpan:
        if not isinstance(value, Mapping) or set(value) != {"start", "end", "text"}:
            raise ValueError("invalid serialized source span")
        return cls(**value)


@dataclass(frozen=True)
class TemporalClaim:
    action_id: str
    relation: str
    anchor_action_id: str
    source_event_id: str
    cue_span: SourceSpan
    action_span: SourceSpan
    anchor_span: SourceSpan
    status: str = "proposed"

    def __post_init__(self) -> None:
        for field in ("action_id", "anchor_action_id", "source_event_id"):
            _identifier(getattr(self, field), field)
        _relation(self.relation)
        if self.action_id == self.anchor_action_id:
            raise ValueError("a strict temporal relation needs two different actions")
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ValueError("invalid temporal claim status")
        if not all(isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("cue, action and anchor need explicit SourceSpan bindings")

    @property
    def spans(self) -> tuple[SourceSpan, SourceSpan, SourceSpan]:
        return self.cue_span, self.action_span, self.anchor_span

    def validate(self, source_text: str) -> TemporalClaim:
        for span in self.spans:
            span.validate(source_text)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "time_basis": "event",
            "action_id": self.action_id, "relation": self.relation,
            "anchor_action_id": self.anchor_action_id,
            "source_event_id": self.source_event_id, "status": self.status,
            "cue_span": self.cue_span.to_dict(),
            "action_span": self.action_span.to_dict(),
            "anchor_span": self.anchor_span.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TemporalClaim:
        fields = {"schema_version", "time_basis", "action_id", "relation",
                  "anchor_action_id", "source_event_id", "status",
                  "cue_span", "action_span", "anchor_span"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("invalid serialized temporal claim")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported temporal claim schema")
        if value["time_basis"] != "event":
            raise ValueError("recording time is not event time")
        args = {k: v for k, v in value.items() if k not in {"schema_version", "time_basis"}}
        for field in ("cue_span", "action_span", "anchor_span"):
            args[field] = SourceSpan.from_dict(args[field])
        return cls(**args)


@dataclass(frozen=True)
class TemporalConsistency:
    consistent: bool
    cycle: tuple[str, ...]
    accepted_claims: int
    ignored_claims: int


@dataclass(frozen=True)
class TemporalAnswer:
    relation: str
    reason: str
    source_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalSelection:
    action_id: str | None
    reason: str
    source_event_ids: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "known" if self.action_id is not None else "unknown"


class TemporalReference:
    """Order supplied, accepted action bindings; never guess a missing binding.

    A cycle makes the entire reference inconsistent and every query unknown.
    Acceptance is a caller's explicit policy decision, not validation of truth.
    Proposed, ambiguous, and rejected claims remain inspectable in ``claims``
    but cannot establish an order. No ingestion timestamp is consulted.
    """

    def __init__(self, claims: Iterable[TemporalClaim], sources: Mapping[str, str]):
        if not isinstance(sources, Mapping):
            raise ValueError("sources must map explicit event IDs to source text")
        self.claims = tuple(claims)
        edges: dict[str, dict[str, set[str]]] = {}
        accepted = 0
        for claim in self.claims:
            if not isinstance(claim, TemporalClaim):
                raise ValueError("expected TemporalClaim objects")
            if claim.source_event_id not in sources:
                raise ValueError(f"missing source event: {claim.source_event_id}")
            claim.validate(sources[claim.source_event_id])
            if claim.status != "accepted":
                continue
            accepted += 1
            early, late = claim.action_id, claim.anchor_action_id
            if claim.relation == AFTER:
                early, late = late, early
            edges.setdefault(early, {}).setdefault(late, set()).add(claim.source_event_id)
            edges.setdefault(late, {})
        # Freeze the graph's iteration order and provenance for deterministic replay.
        self._edges = {a: {b: tuple(sorted(ids)) for b, ids in sorted(row.items())}
                       for a, row in sorted(edges.items())}
        cycle = self._find_cycle()
        self.consistency = TemporalConsistency(not cycle, cycle, accepted,
                                               len(self.claims) - accepted)

    def _find_cycle(self) -> tuple[str, ...]:
        # Iterative DFS supports histories longer than Python's recursion limit.
        seen: set[str] = set()
        for root in self._edges:
            if root in seen:
                continue
            path = [root]
            positions = {root: 0}
            seen.add(root)
            stack = [iter(self._edges[root])]
            while stack:
                child = next(stack[-1], None)
                if child is None:
                    stack.pop()
                    positions.pop(path.pop())
                elif child in positions:
                    return tuple(path[positions[child]:] + [child])
                elif child not in seen:
                    seen.add(child)
                    positions[child] = len(path)
                    path.append(child)
                    stack.append(iter(self._edges[child]))
        return ()

    def _path_sources(self, start: str, goal: str) -> tuple[str, ...] | None:
        queue = deque([start])
        previous: dict[str, str | None] = {start: None}
        while queue:
            node = queue.popleft()
            if node == goal:
                evidence: set[str] = set()
                while previous[node] is not None:
                    parent = previous[node]
                    evidence.update(self._edges[parent][node])
                    node = parent
                return tuple(sorted(evidence))
            for child in self._edges.get(node, {}):
                if child not in previous:
                    previous[child] = node
                    queue.append(child)
        return None

    def relation_between(self, action_id: str, anchor_action_id: str) -> TemporalAnswer:
        _identifier(action_id, "action_id")
        _identifier(anchor_action_id, "anchor_action_id")
        if not self.consistency.consistent:
            return TemporalAnswer(UNKNOWN, "inconsistent")
        if action_id == anchor_action_id:
            return TemporalAnswer(UNKNOWN, "same_action")
        if action_id not in self._edges or anchor_action_id not in self._edges:
            return TemporalAnswer(UNKNOWN, "unbound")
        before = self._path_sources(action_id, anchor_action_id)
        if before is not None:
            return TemporalAnswer(BEFORE, "accepted_path", before)
        after = self._path_sources(anchor_action_id, action_id)
        if after is not None:
            return TemporalAnswer(AFTER, "accepted_path", after)
        return TemporalAnswer(UNKNOWN, "unordered")

    def select_event(self, candidates: Iterable[str], relation: str,
                     anchor_action_id: str) -> TemporalSelection:
        """Select only a unique match when every candidate's order is established.

        One established match plus an unbound candidate is still unknown: the
        unbound candidate could also match. Ties never use input/record order.
        """
        _relation(relation)
        _identifier(anchor_action_id, "anchor_action_id")
        if isinstance(candidates, (str, bytes)):
            raise ValueError("candidates must be an iterable of explicit action IDs")
        candidates = tuple(candidates)
        for candidate in candidates:
            _identifier(candidate, "candidate")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate action IDs must be distinct")
        if not self.consistency.consistent:
            return TemporalSelection(None, "inconsistent")
        if not candidates:
            return TemporalSelection(None, "no_candidates")
        answers = [(a, self.relation_between(a, anchor_action_id)) for a in candidates]
        if any(answer.relation == UNKNOWN for _, answer in answers):
            return TemporalSelection(None, "unbound_or_unordered")
        matches = [(a, answer) for a, answer in answers if answer.relation == relation]
        if len(matches) != 1:
            return TemporalSelection(None, "ambiguous" if matches else "no_match")
        action, _ = matches[0]
        # Include evidence for exclusions as well as the selected event.
        evidence = tuple(sorted({s for _, answer in answers for s in answer.source_event_ids}))
        return TemporalSelection(action, "unique_accepted_match", evidence)


def check_consistency(claims: Iterable[TemporalClaim],
                      sources: Mapping[str, str]) -> TemporalConsistency:
    return TemporalReference(claims, sources).consistency
