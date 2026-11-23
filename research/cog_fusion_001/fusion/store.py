"""Small transactional current-fact store for the COG fusion experiment.

This module enforces storage and continuation contracts, not linguistic truth.
The caller supplies interpreted operations and opaque, non-executed state bytes.
The SQLite transaction is the publication boundary. A request receipt is a
historical result; replaying its id never asks a new current-state question.

Current records are indexed by full (scope, entity, relation) identity. The
archive (receipts, versions and source identities) grows with retained history.
There is no claim of unbounded information in bounded neural state.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterable, Sequence
import uuid


Key = tuple[str, str, str]
APPLICATION_ID = 0x43464731
MAX_BLOB_BYTES = 64 * 1024 * 1024


class StoreError(RuntimeError):
    pass


class ConflictError(StoreError):
    """The proposed continuation or request identity is stale/conflicting."""


class CapacityError(StoreError):
    """The declared current-record capacity would be exceeded."""


class IntegrityError(StoreError):
    """Stored bytes, interpretation identity or a packet failed validation."""


class InjectedFailure(StoreError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: str, label: str = "identifier") -> str:
    if (not isinstance(value, str) or not value or len(value) > 512
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{label} must be 1..512 non-control characters")
    return value


def _key(value: Key | Sequence[str]) -> Key:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("key must be (scope, entity, relation)")
    return tuple(_identifier(x, "key component") for x in value)  # type: ignore[return-value]


def _bytes(value: bytes | None) -> bytes | None:
    if value is not None and (not isinstance(value, bytes) or len(value) > MAX_BLOB_BYTES):
        raise ValueError("continuation must be bytes within the 64 MiB limit")
    return value


def _blob_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Operation:
    kind: str
    key: Key
    value: Any = None
    source: str = ""
    copy_from: Key | None = None


@dataclass(frozen=True)
class Dependency:
    key: Key
    token: str
    version: int
    status: str


@dataclass(frozen=True)
class EvidencePacket:
    branch_id: str
    model_id: str
    schema_id: str
    key: Key
    follow_relation: str | None
    dependencies: tuple[Dependency, ...]
    values: tuple[Any, ...]
    status: str
    value: Any
    read_count: int
    revision: int
    pending: tuple[str, ...] = ()
    packet_digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    """Bounded selection diagnostic; scoring cost belongs to the caller."""
    status: str
    selected_key: Key | None
    selected_value: Any
    dependencies: tuple[Dependency, ...]
    candidates_inspected: int


_SCHEMA = """
CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runtime (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL,
    neural_state BLOB NOT NULL, rng_state BLOB NOT NULL,
    neural_digest TEXT NOT NULL, rng_digest TEXT NOT NULL, last_request TEXT);
CREATE TABLE sources (
    source TEXT PRIMARY KEY, generation INTEGER NOT NULL, active INTEGER NOT NULL);
CREATE TABLE records (
    key TEXT PRIMARY KEY, scope TEXT NOT NULL, entity TEXT NOT NULL, relation TEXT NOT NULL,
    value TEXT, source TEXT NOT NULL, origins TEXT NOT NULL,
    version INTEGER NOT NULL, incarnation TEXT NOT NULL, kind TEXT NOT NULL,
    present INTEGER NOT NULL, reason TEXT NOT NULL, digest TEXT NOT NULL);
CREATE INDEX records_scope ON records(scope, present);
CREATE TABLE pending (
    pending_id TEXT PRIMARY KEY, scope TEXT, key TEXT,
    fingerprint TEXT NOT NULL, target TEXT NOT NULL);
CREATE INDEX pending_scope ON pending(scope);
CREATE TABLE pending_versions (target TEXT PRIMARY KEY, generation INTEGER NOT NULL);
CREATE TABLE pending_history (
    pending_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
    target TEXT NOT NULL, completed INTEGER NOT NULL, resolved_request TEXT);
CREATE TABLE receipts (
    request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
    receipt TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE events (
    event INTEGER PRIMARY KEY AUTOINCREMENT,
    revision INTEGER NOT NULL, request_id TEXT NOT NULL, operation TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str | Path, model_id: str, *,
                 schema_id: str = "cog-fusion-v1", branch_id: str | None = None,
                 capacity: int | None = None) -> None:
        self.path = str(path)
        self.model_id = _identifier(model_id, "model_id")
        self.schema_id = _identifier(schema_id, "schema_id")
        if branch_id is not None:
            _identifier(branch_id, "branch_id")
        if capacity is not None and (type(capacity) is not int or capacity < 1):
            raise ValueError("capacity must be a positive integer")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=20000")
        application_id = self.db.execute("PRAGMA application_id").fetchone()[0]
        tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not tables and application_id == 0:
            self.db.executescript("BEGIN IMMEDIATE;" + _SCHEMA)
            try:
                self.branch_id = branch_id or uuid.uuid4().hex
                self.capacity = 4096 if capacity is None else capacity
                self.db.executemany("INSERT INTO metadata VALUES (?,?)", (
                    ("branch_id", self.branch_id), ("model_id", self.model_id),
                    ("schema_id", self.schema_id), ("capacity", str(self.capacity)),
                    ("format", "2")))
                self.db.execute("INSERT INTO runtime VALUES (1,0,?,?,?,?,NULL)",
                                (b"", b"", _blob_digest(b""), _blob_digest(b"")))
                self.db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                self.db.commit()
            except BaseException:
                self.db.rollback()
                self.db.close()
                raise
        elif application_id != APPLICATION_ID:
            self.db.close()
            raise IntegrityError("database is not a COG fusion store")
        else:
            metadata = dict(self.db.execute("SELECT name,value FROM metadata"))
            if (metadata.get("model_id") != self.model_id or
                    metadata.get("schema_id") != self.schema_id or metadata.get("format") != "2"):
                self.db.close()
                raise IntegrityError("model or schema interpretation differs from saved state")
            self.branch_id = metadata["branch_id"]
            self.capacity = int(metadata["capacity"])
            if ((branch_id is not None and branch_id != self.branch_id)
                    or (capacity is not None and capacity != self.capacity)):
                self.db.close()
                raise IntegrityError("branch or capacity differs from saved contract")
        self._observed_revision = self.db.execute("SELECT revision FROM runtime").fetchone()[0]
        self.counters = {"record_reads": 0, "validation_reads": 0,
                         "commits": 0, "receipt_replays": 0, "source_checks": 0}

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _identity(self) -> None:
        observed = dict(self.db.execute("SELECT name,value FROM metadata"))
        expected = {"branch_id": self.branch_id, "model_id": self.model_id,
                    "schema_id": self.schema_id, "capacity": str(self.capacity), "format": "2"}
        if observed != expected:
            raise IntegrityError("store interpretation changed while connection was open")

    @contextmanager
    def _transaction(self, *, write: bool = False):
        self.db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            self._identity()
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def _revision(self) -> int:
        return self.db.execute("SELECT revision FROM runtime WHERE singleton=1").fetchone()[0]

    def _cas(self, expected: int | None) -> int:
        expected = self._observed_revision if expected is None else expected
        if type(expected) is not int or expected < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        current = self._revision()
        if expected != current:
            raise ConflictError(f"stale continuation: expected revision {expected}, current {current}")
        return current

    @staticmethod
    def _record_payload(row) -> dict:
        return {k: row[k] for k in ("key", "value", "source", "origins", "version",
                                    "incarnation", "kind", "present", "reason")}

    def _record(self, key: Key, *, validation: bool = False) -> tuple[Dependency, Any, tuple[str, ...]]:
        self.counters["validation_reads" if validation else "record_reads"] += 1
        encoded = canonical(key)
        row = self.db.execute("SELECT * FROM records WHERE key=?", (encoded,)).fetchone()
        pending = tuple(x[0] for x in self.db.execute(
            "SELECT pending_id FROM pending WHERE scope IS NULL OR "
            "(scope=? AND (key IS NULL OR key=?)) ORDER BY pending_id", (key[0], encoded)))
        targets = ("*", "scope:" + key[0], "key:" + encoded)
        generations = [self.db.execute("SELECT generation FROM pending_versions WHERE target=?", (t,))
                       .fetchone() for t in targets]
        pending_generations = [x[0] if x else 0 for x in generations]
        if row is None:
            token = digest((self.branch_id, key, "never-present", pending_generations))
            return Dependency(key, token, 0, "pending" if pending else "missing"), None, pending
        if digest(self._record_payload(row)) != row["digest"]:
            raise IntegrityError("record checksum mismatch")
        origins = json.loads(row["origins"])
        source_states = []
        valid_source = True
        for source, generation in origins:
            self.counters["source_checks"] += 1
            current = self.db.execute("SELECT generation,active FROM sources WHERE source=?", (source,)).fetchone()
            if current is None:
                raise IntegrityError("record refers to a missing provenance identity")
            source_states.append((source, current["generation"], current["active"]))
            valid_source &= current["generation"] == generation and bool(current["active"])
        status = "ok" if row["present"] and valid_source else row["reason"] or "deleted"
        if not valid_source:
            status = "retracted"
        if pending:
            status = "pending"
        token = digest((self.branch_id, key, row["digest"], source_states, pending_generations))
        value = json.loads(row["value"]) if row["present"] and row["value"] is not None else None
        return Dependency(key, token, row["version"], status), value, pending

    def _packet(self, key: Key, follow_relation: str | None, *, validation: bool = False) -> EvidencePacket:
        dep, value, pending = self._record(key, validation=validation)
        deps, values, pending_ids = [dep], [], list(pending)
        status = dep.status
        if status == "ok":
            values.append(value)
            if follow_relation is not None:
                if not isinstance(value, str) or not value:
                    status = "invalid_link"
                    value = None
                else:
                    second_key = _key((key[0], value, follow_relation))
                    dep2, value, pending2 = self._record(second_key, validation=validation)
                    deps.append(dep2)
                    pending_ids.extend(pending2)
                    status = dep2.status
                    if status == "ok":
                        values.append(value)
        if status != "ok":
            value = None
        packet = EvidencePacket(self.branch_id, self.model_id, self.schema_id, key,
                                follow_relation, tuple(deps), tuple(values), status, value,
                                len(deps), self._revision(), tuple(sorted(set(pending_ids))))
        return replace(packet, packet_digest=digest(packet.to_dict()))

    def resolve(self, key: Key, follow_relation: str | None = None) -> EvidencePacket:
        """Read one exact record, or two where hop two uses hop one's real value."""
        key = _key(key)
        if follow_relation is not None:
            _identifier(follow_relation, "follow_relation")
        with self._transaction():
            return self._packet(key, follow_relation)

    def _validate_packet(self, packet: EvidencePacket) -> bool:
        if not isinstance(packet, EvidencePacket):
            return False
        if (packet.branch_id, packet.model_id, packet.schema_id) != (
                self.branch_id, self.model_id, self.schema_id):
            return False
        if packet.packet_digest != digest(replace(packet, packet_digest="").to_dict()):
            return False
        fresh = self._packet(packet.key, packet.follow_relation, validation=True)
        # Revision is not a dependency: an unrelated edit preserves typed results.
        return all(getattr(packet, name) == getattr(fresh, name) for name in (
            "dependencies", "values", "status", "value", "read_count", "pending"))

    def validate_packet(self, packet: EvidencePacket) -> bool:
        with self._transaction():
            return self._validate_packet(packet)

    def select_candidates(self, keys: Sequence[Key], scores: Sequence[float], *,
                          max_candidates: int = 4, tie_tolerance: float = 1e-8) -> CandidateResult:
        """Inspect at most four supplied candidates; never scan the archive.

        Candidate generation and learned scoring are outside this method. All
        attempted candidates, including missing ones, become dependencies.
        Over-budget requests fail before any record access; equal top scores
        fail explicitly. This is an executor diagnostic, not a learned reader.
        """
        if type(max_candidates) is not int or not 1 <= max_candidates <= 4:
            raise ValueError("candidate budget must be in 1..4")
        if len(keys) != len(scores) or not keys:
            raise ValueError("provide a score for each nonempty candidate list")
        if len(keys) > max_candidates:
            return CandidateResult("overflow", None, None, (), 0)
        keys = tuple(_key(k) for k in keys)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate candidates are not separate evidence")
        scores = tuple(float(x) for x in scores)
        if not all(math.isfinite(x) for x in scores) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError("scores and tie tolerance must be finite")
        with self._transaction():
            rows = [self._record(k) for k in keys]
            deps = tuple(row[0] for row in rows)
            if any(d.status == "pending" for d in deps):
                return CandidateResult("pending", None, None, deps, len(rows))
            eligible = [i for i, d in enumerate(deps) if d.status == "ok"]
            if not eligible:
                return CandidateResult("missing", None, None, deps, len(rows))
            ordered = sorted(eligible, key=lambda i: scores[i], reverse=True)
            if len(ordered) > 1 and scores[ordered[0]] - scores[ordered[1]] <= tie_tolerance:
                return CandidateResult("tie", None, None, deps, len(rows))
            selected = ordered[0]
            return CandidateResult("ok", keys[selected], rows[selected][1], deps, len(rows))

    def _source(self, source: str) -> tuple[str, int]:
        _identifier(source, "source")
        row = self.db.execute("SELECT generation,active FROM sources WHERE source=?", (source,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO sources VALUES (?,1,1)", (source,))
            return source, 1
        if not row["active"]:
            raise ConflictError("an erased source identity cannot be resurrected")
        return source, row["generation"]

    def _write_record(self, op: Operation, request_id: str, revision: int) -> None:
        key = _key(op.key)
        if op.kind not in {"set", "link", "delete", "copy"}:
            raise ValueError("unsupported operation kind")
        if op.kind != "copy" and op.copy_from is not None:
            raise ValueError("copy_from is valid only for copy")
        encoded = canonical(key)
        old = self.db.execute("SELECT * FROM records WHERE key=?", (encoded,)).fetchone()
        if old is not None and digest(self._record_payload(old)) != old["digest"]:
            raise IntegrityError("cannot overwrite a corrupted record")
        source = op.source or request_id
        origin = self._source(source)
        origins = [origin]
        value = op.value
        if op.kind == "link":
            _identifier(value, "link target entity")
        elif op.kind == "copy":
            if op.copy_from is None:
                raise ValueError("copy requires copy_from")
            source_key = _key(op.copy_from)
            dep, value, _ = self._record(source_key)
            if dep.status != "ok":
                raise ConflictError("copy source is unavailable or pending")
            source_row = self.db.execute("SELECT origins FROM records WHERE key=?", (canonical(source_key),)).fetchone()
            origins += [(s, g) for s, g in json.loads(source_row["origins"])]
        present = int(op.kind != "delete")
        if present and (old is None or not old["present"]):
            used = self.db.execute("SELECT COUNT(*) FROM records WHERE present=1").fetchone()[0]
            if used >= self.capacity:
                raise CapacityError("current-record capacity exhausted; no implicit eviction")
        payload = dict(key=encoded, value=canonical(value) if present else None,
                       source=source, origins=canonical(sorted(set(origins))),
                       version=1 if old is None else old["version"] + 1,
                       incarnation=uuid.uuid4().hex, kind=op.kind, present=present,
                       reason="" if present else "deleted")
        columns = (encoded, *key, payload["value"], source, payload["origins"],
                   payload["version"], payload["incarnation"], op.kind, present,
                   payload["reason"], digest(payload))
        self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", columns)
        self.db.execute("INSERT INTO events(revision,request_id,operation) VALUES (?,?,?)",
                        (revision, request_id, canonical(asdict(op))))

    def _invalidate_source(self, source: str, request_id: str, revision: int) -> None:
        _identifier(source, "source")
        row = self.db.execute("SELECT generation,active FROM sources WHERE source=?", (source,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO sources VALUES (?,1,0)", (source,))
        elif row["active"]:
            self.db.execute("UPDATE sources SET generation=generation+1,active=0 WHERE source=?", (source,))
        # Charged source-erasure work may inspect current provenance. Token reads
        # never do this scan. Tombstoning releases active capacity without losing
        # the key's incarnation history or silently restoring an older fact.
        for current in self.db.execute("SELECT * FROM records WHERE present=1").fetchall():
            if any(s == source for s, _ in json.loads(current["origins"])):
                if digest(self._record_payload(current)) != current["digest"]:
                    raise IntegrityError("corrupt record during source erasure")
                payload = self._record_payload(current)
                payload.update(present=0, value=None, reason="retracted",
                               version=payload["version"] + 1, incarnation=uuid.uuid4().hex)
                self.db.execute("UPDATE records SET value=NULL,present=0,reason='retracted',"
                                "version=?,incarnation=?,digest=? WHERE key=?",
                                (payload["version"], payload["incarnation"], digest(payload), current["key"]))
        self.db.execute("INSERT INTO events(revision,request_id,operation) VALUES (?,?,?)",
                        (revision, request_id, canonical({"erase_source": source})))

    def _bump_pending(self, target: str) -> None:
        self.db.execute("INSERT INTO pending_versions VALUES (?,1) ON CONFLICT(target) "
                        "DO UPDATE SET generation=generation+1", (target,))

    def mark_pending(self, pending_id: str, scope: str | None = None, *,
                     key: Key | None = None, payload: Any = None,
                     expected_revision: int | None = None) -> dict:
        """Persist an unresolved intent before attempting its factual publication.

        Unknown affected scope is represented by ``scope=None`` and blocks all
        queries. This method deliberately does not advance the neural state.
        """
        _identifier(pending_id, "pending_id")
        if scope is not None:
            _identifier(scope, "scope")
        if key is not None:
            key = _key(key)
            if scope is not None and scope != key[0]:
                raise ValueError("pending scope and key disagree")
            scope = key[0]
        encoded = canonical(key) if key is not None else None
        target = "key:" + encoded if encoded else "scope:" + scope if scope else "*"
        fingerprint = digest((scope, key, payload))
        with self._transaction(write=True):
            old = self.db.execute("SELECT * FROM pending_history WHERE pending_id=?", (pending_id,)).fetchone()
            if old is not None:
                if old["fingerprint"] != fingerprint:
                    raise ConflictError("pending id reused with another intent")
                return {"pending_id": pending_id, "revision": self._revision(), "replayed": True,
                        "resolved": bool(old["completed"])}
            revision = self._cas(expected_revision) + 1
            self.db.execute("INSERT INTO pending VALUES (?,?,?,?,?)",
                            (pending_id, scope, encoded, fingerprint, target))
            self.db.execute("INSERT INTO pending_history VALUES (?,?,?,0,NULL)",
                            (pending_id, fingerprint, target))
            self._bump_pending(target)
            self.db.execute("UPDATE runtime SET revision=? WHERE singleton=1", (revision,))
        self._observed_revision = revision
        return {"pending_id": pending_id, "revision": revision, "replayed": False, "resolved": False}

    def _clear_pending(self, pending_id: str, request_id: str) -> None:
        row = self.db.execute("SELECT target FROM pending WHERE pending_id=?", (pending_id,)).fetchone()
        if row is None:
            raise ConflictError("pending intent does not exist")
        self._bump_pending(row["target"])
        self.db.execute("DELETE FROM pending WHERE pending_id=?", (pending_id,))
        self.db.execute("UPDATE pending_history SET completed=1,resolved_request=? WHERE pending_id=?",
                        (request_id, pending_id))

    def _lookup_receipt(self, request_id: str, fingerprint: str) -> dict | None:
        previous = self.db.execute("SELECT * FROM receipts WHERE request_id=?", (request_id,)).fetchone()
        if previous is None:
            return None
        if previous["fingerprint"] != fingerprint:
            raise ConflictError("request id reused for different input")
        if _blob_digest(previous["receipt"].encode()) != previous["digest"]:
            raise IntegrityError("receipt checksum mismatch")
        receipt = json.loads(previous["receipt"])
        receipt["replayed"] = True
        return receipt

    def lookup_receipt(self, request_id: str, payload: Any) -> dict | None:
        """Check an exact retry before advancing neural state or consuming RNG."""
        _identifier(request_id, "request_id")
        with self._transaction():
            receipt = self._lookup_receipt(request_id, digest(payload))
        if receipt is not None:
            self.counters["receipt_replays"] += 1
        return receipt

    @staticmethod
    def _fail(failpoint: str | Callable[[str], None] | None, stage: str) -> None:
        if callable(failpoint):
            failpoint(stage)
        elif failpoint == stage:
            raise InjectedFailure(stage)

    def commit_request(self, request_id: str, payload: Any, *,
                       operations: Iterable[Operation] = (), neural_state: bytes | None = None,
                       rng_state: bytes | None = None, reply: Any = None,
                       packet: EvidencePacket | None = None,
                       clear_pending: Iterable[str] = (), invalidate_sources: Iterable[str] = (),
                       expected_revision: int | None = None,
                       failpoint: str | Callable[[str], None] | None = None) -> dict:
        """Atomically publish facts, continuation, RNG, dependencies and reply.

        The opaque input payload is the idempotency fingerprint. Results and
        stochastic state are not part of it: a lost-response retry returns the
        already committed result. A new question requires a new request id.
        """
        _identifier(request_id, "request_id")
        fingerprint = digest(payload)
        _bytes(neural_state); _bytes(rng_state)
        operations = tuple(operations)
        clear_pending, invalidate_sources = tuple(clear_pending), tuple(invalidate_sources)
        canonical(reply)
        with self._transaction(write=True):
            previous = self._lookup_receipt(request_id, fingerprint)
            if previous is not None:
                self.counters["receipt_replays"] += 1
                return previous
            revision = self._cas(expected_revision) + 1
            self._fail(failpoint, "before_operations")
            for operation in operations:
                if not isinstance(operation, Operation):
                    raise ValueError("operations must be Operation instances")
                self._write_record(operation, request_id, revision)
            for source in invalidate_sources:
                self._invalidate_source(source, request_id, revision)
            for pending_id in clear_pending:
                self._clear_pending(_identifier(pending_id, "pending_id"), request_id)
            self._fail(failpoint, "after_operations")
            if packet is not None and not self._validate_packet(packet):
                raise ConflictError("reply evidence changed before publication")
            runtime = self.db.execute("SELECT * FROM runtime WHERE singleton=1").fetchone()
            if (_blob_digest(runtime["neural_state"]) != runtime["neural_digest"]
                    or _blob_digest(runtime["rng_state"]) != runtime["rng_digest"]):
                raise IntegrityError("previous continuation checksum mismatch")
            neural_state = runtime["neural_state"] if neural_state is None else neural_state
            rng_state = runtime["rng_state"] if rng_state is None else rng_state
            neural_digest, rng_digest = _blob_digest(neural_state), _blob_digest(rng_state)
            receipt = {"request_id": request_id, "input_digest": fingerprint,
                       "revision": revision, "branch_id": self.branch_id,
                       "model_id": self.model_id, "schema_id": self.schema_id,
                       "reply": reply, "packet": None if packet is None else packet.to_dict(),
                       "neural_digest": neural_digest, "rng_digest": rng_digest,
                       "pending_ids": [x[0] for x in self.db.execute("SELECT pending_id FROM pending ORDER BY pending_id")],
                       "replayed": False, "historical_receipt": True}
            serialized = canonical(receipt)
            # First response and replay expose exactly the same JSON value.
            receipt = json.loads(serialized)
            self.db.execute("INSERT INTO receipts VALUES (?,?,?,?)",
                            (request_id, fingerprint, serialized, _blob_digest(serialized.encode())))
            self.db.execute("UPDATE runtime SET revision=?,neural_state=?,rng_state=?,"
                            "neural_digest=?,rng_digest=?,last_request=? WHERE singleton=1",
                            (revision, neural_state, rng_state, neural_digest, rng_digest, request_id))
            self._fail(failpoint, "before_commit")
        self._observed_revision = revision
        self.counters["commits"] += 1
        self._fail(failpoint, "after_commit")
        return receipt

    def snapshot_state(self, *, expected_revision: int | None = None) -> dict:
        with self._transaction():
            revision = self._cas(expected_revision)
            row = dict(self.db.execute("SELECT * FROM runtime WHERE singleton=1").fetchone())
            if (_blob_digest(row["neural_state"]) != row["neural_digest"]
                    or _blob_digest(row["rng_state"]) != row["rng_digest"]):
                raise IntegrityError("continuation checksum mismatch")
            if row["last_request"] is not None:
                stored = self.db.execute("SELECT receipt,digest FROM receipts WHERE request_id=?",
                                         (row["last_request"],)).fetchone()
                if stored is None or _blob_digest(stored["receipt"].encode()) != stored["digest"]:
                    raise IntegrityError("continuation receipt is missing or corrupt")
                receipt = json.loads(stored["receipt"])
                if (receipt["request_id"] != row["last_request"]
                        or receipt["neural_digest"] != row["neural_digest"]
                        or receipt["rng_digest"] != row["rng_digest"]
                        or receipt["revision"] > revision
                        or receipt["model_id"] != self.model_id
                        or receipt["schema_id"] != self.schema_id):
                    raise IntegrityError("continuation is not bound to its committed receipt")
            row.pop("singleton")
            row.update(branch_id=self.branch_id, model_id=self.model_id,
                       schema_id=self.schema_id,
                       pending_ids=[x[0] for x in self.db.execute("SELECT pending_id FROM pending ORDER BY pending_id")])
        self._observed_revision = revision
        return row

    def refresh_snapshot(self) -> dict:
        """Explicitly discard the caller's stale continuation and load current state."""
        with self._transaction():
            revision = self._revision()
        return self.snapshot_state(expected_revision=revision)

    def invalidate_source(self, source: str, *, request_id: str) -> dict:
        return self.commit_request(request_id, {"invalidate_source": source}, invalidate_sources=(source,))

    def clear_pending(self, pending_id: str, *, request_id: str) -> dict:
        return self.commit_request(request_id, {"clear_pending": pending_id}, clear_pending=(pending_id,))

    def current_records(self) -> dict[Key, Any]:
        """Diagnostic export; never used by bounded resolution."""
        with self._transaction():
            result = {}
            for row in self.db.execute("SELECT scope,entity,relation FROM records WHERE present=1").fetchall():
                key = tuple(row)
                dep, value, _ = self._record(key)
                if dep.status == "ok":
                    result[key] = value
            return result

    def fork(self, destination: str | Path, *, branch_id: str | None = None) -> "Store":
        destination = Path(destination)
        if any(Path(str(destination) + suffix).exists() for suffix in ("", "-wal", "-shm", "-journal")):
            raise FileExistsError("fork target or SQLite sidecar already exists")
        branch_id = _identifier(branch_id or uuid.uuid4().hex, "branch_id")
        if branch_id == self.branch_id:
            raise ValueError("fork requires a different branch identity")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination, isolation_level=None)
        try:
            self.db.backup(target)
            target.execute("BEGIN IMMEDIATE")
            target.execute("UPDATE metadata SET value=? WHERE name='branch_id'", (branch_id,))
            target.commit()
        finally:
            target.close()
        return Store(destination, self.model_id, schema_id=self.schema_id,
                     branch_id=branch_id, capacity=self.capacity)
