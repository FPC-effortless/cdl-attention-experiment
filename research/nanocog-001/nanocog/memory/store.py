"""Transactional local memory with scoped identities and non-executable data.

The host authenticates ``principal_id``. Namespaces isolate queries, not operating
system users. SQLite is local persistent storage; backups need a separate disk.
"""
from __future__ import annotations

from contextlib import contextmanager, closing
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import time
from typing import Any
import unicodedata

SCHEMA_VERSION = 2
APPLICATION_ID = 0x434F474D  # COGM; do not adopt unrelated SQLite files.
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
CHECKPOINT_OUTPUTS = 128
CHECKPOINT_OUTPUT_BYTES = 32 * 1024 * 1024
_WORD_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class ConflictError(RuntimeError):
    """An idempotency key or expected revision conflicts with committed state."""


class IntegrityError(RuntimeError):
    """Stored bytes, schema or provenance do not satisfy the storage contract."""


class DeletedError(ConflictError):
    """An erased identifier cannot be resurrected by retrying an old request."""


class ReceiptExpiredError(ConflictError):
    """The request was applied, but its bounded output cache has expired."""


def default_path() -> Path:
    override = os.environ.get("COG_MEMORY_PATH")
    if override:
        return Path(override).expanduser()
    configured = os.environ.get("XDG_STATE_HOME")
    root = Path(configured) if configured and Path(configured).is_absolute() else Path.home() / ".local" / "state"
    return root / "nanocog" / "memory.sqlite3"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be a nonempty string of at most 512 characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


def _json(value: Any, limit: int = MAX_JSON_BYTES) -> str:
    try:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("memory accepts finite JSON data only") from exc
    if len(result.encode("utf-8")) > limit:
        raise ValueError("memory payload exceeds byte limit")
    return result


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_tokens(text: str) -> list[str]:
    # FTS remains the candidate matcher. Rank using local text, never corpus
    # statistics from other principals. These are unicode61's usual boundaries.
    folded = text.casefold()
    # NFD is the identity on ASCII and its combining classes are all zero.
    # Avoid scanning those characters in Python for every matching record.
    if not folded.isascii():
        folded = ''.join(c for c in unicodedata.normalize('NFD', folded)
                         if not unicodedata.combining(c))
    return _WORD_TOKEN_RE.findall(folded)


@lru_cache(maxsize=64)
def _rank_terms(query: str) -> frozenset[str]:
    return frozenset(_word_tokens(query))


def _lexical_score(content: str, query: str) -> float:
    words = _word_tokens(content)
    overlap = len(set(words).intersection(_rank_terms(query)))
    # Distinct matching terms outrank fewer terms; density breaks coverage ties.
    return overlap + overlap / (1 + len(words))


def _checkpoint_identity(row) -> str:
    try:
        manifest = json.loads(row['manifest'])
        if not isinstance(manifest, dict) or _json(manifest) != row['manifest']:
            raise ValueError('noncanonical checkpoint manifest')
        identity = dict(principal=row['principal'], namespace=row['namespace'],
                        session=row['session'], request=row['request'],
                        request_hash=row['request_hash'], revision=row['revision'],
                        manifest=manifest)
        if 'payload' in row.keys():
            identity.update(payload_digest=row['digest'], result_digest=row['result_digest'])
        else:
            identity['result_digest'] = row['digest']
        return _digest(_json(identity))
    except (KeyError, ValueError, TypeError, RecursionError) as exc:
        raise IntegrityError('invalid checkpoint identity metadata') from exc


def _decode(text: str, digest: str) -> Any:
    if not isinstance(text, str) or _digest(text) != digest:
        raise IntegrityError("stored payload checksum mismatch")
    try:
        value = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if _json(value) != text:
            raise ValueError("noncanonical JSON")
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise IntegrityError("invalid stored JSON") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _revision(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected_revision must be a nonnegative integer")
    return value


def _prepare_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not os.path.lexists(path):
        try:
            _unused_database_path(path)
        except FileExistsError:
            # A peer initializer can create the live DB between these checks.
            # Still refuse an absent database with stale SQLite sidecars.
            if not os.path.lexists(path):
                raise
    # Never follow an existing final-component symlink. Parent directories belong
    # to the caller; this local API is not a sandbox against that same OS user.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("memory path must be a regular file")
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _unused_database_path(path: Path) -> None:
    # A stale WAL can override a freshly copied main file. A restore target must
    # have no database OR journal sidecars and must not be opened concurrently.
    if any(os.path.lexists(str(path)+suffix) for suffix in ("","-wal","-shm","-journal")):
        raise FileExistsError("database destination or SQLite sidecars already exist")


def _patched_wal(version: tuple[int, ...]) -> bool:
    # SQLite's 2026 WAL-reset fix: https://sqlite.org/wal.html#walreset
    return (version >= (3,51,3) or (3,50,7) <= version < (3,51,0)
            or (3,44,6) <= version < (3,45,0))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_file(temp: Path, destination: Path) -> None:
    # link is atomic and refuses an existing destination. Never clobber a live DB.
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    os.link(temp, destination)
    temp.unlink()
    _fsync_dir(destination.parent)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".cog-memory-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _publish_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


_SCHEMA = (
    "CREATE TABLE scopes (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "erasure_generation INTEGER NOT NULL DEFAULT 0 CHECK(erasure_generation>=0), "
    "PRIMARY KEY(principal, namespace))",
    "CREATE TABLE sessions (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "session TEXT NOT NULL, created TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(principal,namespace,session))",
    "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, principal TEXT NOT NULL, "
    "namespace TEXT NOT NULL, event TEXT NOT NULL, session TEXT NOT NULL, role TEXT NOT NULL, "
    "content TEXT, digest TEXT NOT NULL, created TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, "
    "UNIQUE(principal,namespace,event))",
    "CREATE INDEX messages_scope ON messages(principal,namespace,deleted,seq)",
    "CREATE INDEX messages_session ON messages(principal,namespace,session,seq)",
    "CREATE TABLE message_sources (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "event TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(principal,namespace,event,source))",
    "CREATE INDEX message_dependencies ON message_sources(principal,namespace,source)",
    "CREATE VIRTUAL TABLE message_search USING fts5(content, tokenize='unicode61')",
    "CREATE TABLE proposals (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "proposal TEXT NOT NULL, key TEXT NOT NULL, value TEXT, digest TEXT NOT NULL, "
    "source TEXT NOT NULL, base_revision INTEGER NOT NULL, status TEXT NOT NULL, "
    "created TEXT NOT NULL, PRIMARY KEY(principal,namespace,proposal))",
    "CREATE INDEX proposal_source ON proposals(principal,namespace,source)",
    "CREATE TABLE facts (principal TEXT NOT NULL, namespace TEXT NOT NULL, key TEXT NOT NULL, "
    "revision INTEGER NOT NULL, value TEXT, digest TEXT NOT NULL, source TEXT, "
    "proposal TEXT, authority TEXT, changed TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(principal,namespace,key))",
    "CREATE INDEX facts_source ON facts(principal,namespace,source)",
    "CREATE VIRTUAL TABLE fact_search USING fts5(content, tokenize='unicode61')",
    "CREATE TABLE fact_history (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "key TEXT NOT NULL, revision INTEGER NOT NULL, value TEXT, digest TEXT NOT NULL, "
    "source TEXT, proposal TEXT, authority TEXT, changed TEXT NOT NULL, deleted INTEGER NOT NULL, "
    "PRIMARY KEY(principal,namespace,key,revision))",
    "CREATE INDEX fact_history_source ON fact_history(principal,namespace,source)",
    "CREATE TABLE checkpoints (principal TEXT NOT NULL, namespace TEXT NOT NULL, session TEXT NOT NULL, "
    "revision INTEGER NOT NULL, manifest TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL, "
    "request TEXT NOT NULL, request_hash TEXT NOT NULL, result TEXT NOT NULL, result_digest TEXT NOT NULL, "
    "identity_digest TEXT NOT NULL, "
    "PRIMARY KEY(principal,namespace,session))",
    "CREATE TABLE checkpoint_requests (principal TEXT NOT NULL, namespace TEXT NOT NULL, "
    "session TEXT NOT NULL, request TEXT NOT NULL, request_hash TEXT NOT NULL, revision INTEGER NOT NULL, "
    "manifest TEXT NOT NULL, result TEXT, digest TEXT NOT NULL, identity_digest TEXT NOT NULL, "
    "PRIMARY KEY(principal,namespace,session,request))",
    "CREATE INDEX checkpoint_revision ON checkpoint_requests(principal,namespace,session,revision)",
    "CREATE INDEX checkpoint_live_outputs ON checkpoint_requests(principal,namespace,session,revision) WHERE result IS NOT NULL",
    "CREATE TABLE erasures (principal TEXT NOT NULL, namespace TEXT NOT NULL, event TEXT NOT NULL, "
    "erased TEXT NOT NULL, PRIMARY KEY(principal,namespace,event))",
)


class MemoryStore:
    """One trusted principal/project view over a local durable database.

    Open one connection per thread/process. All mutations are immediate durable
    transactions. Callers never need a separate save operation.
    """
    def __init__(self, path: str | Path | None = None, *, principal_id: str,
                 namespace: str = "default", timeout: float = 10.0):
        self.principal_id = _identifier(principal_id, "principal_id")
        self.namespace = _identifier(namespace, "namespace")
        self.scope = (self.principal_id, self.namespace)
        if sqlite3.sqlite_version_info < (3,42,0):
            raise RuntimeError("SQLite >=3.42 is required for search-index erasure")
        self.path = Path(path if path is not None else default_path()).expanduser().absolute()
        if str(path) == ":memory:":
            raise ValueError("durable memory requires a filesystem path")
        _prepare_file(self.path)
        self.db = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.create_function('cog_lexical_score', 2, _lexical_score, deterministic=True)
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA secure_delete=ON")
            with self._transaction():
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                app = self.db.execute("PRAGMA application_id").fetchone()[0]
                existing = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if version == 0 and app == 0 and not existing:
                    for statement in _SCHEMA:
                        self.db.execute(statement)
                    self.db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif app == APPLICATION_ID and version == 1:
                    self._migrate_v1()
                    self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif app != APPLICATION_ID or version != SCHEMA_VERSION:
                    raise IntegrityError("unsupported memory database/schema; explicit migration required")
                # Ordinary secure_delete does not erase stale FTS5 postings.
                self.db.execute("INSERT INTO message_search(message_search,rank) VALUES('secure-delete',1)")
                self.db.execute("INSERT INTO fact_search(fact_search,rank) VALUES('secure-delete',1)")
                self.db.execute("INSERT OR IGNORE INTO scopes(principal,namespace) VALUES (?,?)", self.scope)
            requested = "WAL" if _patched_wal(sqlite3.sqlite_version_info) else "DELETE"
            # Journal changes can fail immediately with BUSY even when SQLite's
            # busy_timeout is configured. Peers may still be finishing their
            # first schema transaction. Retry only lock contention, outside any
            # transaction, and preserve the caller's finite wait budget.
            deadline = time.monotonic() + max(0.0, timeout)
            configured_timeout = self.db.execute("PRAGMA busy_timeout").fetchone()[0]
            self.db.execute("PRAGMA busy_timeout=0")
            try:
                while True:
                    try:
                        mode = self.db.execute("PRAGMA journal_mode").fetchone()[0]
                        if mode.upper() != requested:
                            mode = self.db.execute(f"PRAGMA journal_mode={requested}").fetchone()[0]
                        break
                    except sqlite3.OperationalError as exc:
                        code = getattr(exc, 'sqlite_errorcode', None)
                        remaining = deadline - time.monotonic()
                        if code is None or (code & 255) not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or remaining <= 0:
                            raise
                        time.sleep(min(0.01, remaining))
            finally:
                self.db.execute(f"PRAGMA busy_timeout={configured_timeout}")
            if mode.upper() != requested:
                raise IntegrityError("durable journaling unavailable")
            self.journal_mode=mode.lower()
            self.db.execute("PRAGMA synchronous=FULL")
            _fsync_dir(self.path.parent)
        except BaseException:
            self.db.close()
            raise

    def _migrate_v1(self):
        """Atomically migrate recognized v1 storage, preserving user records.

        Legacy content checksums are verified before sealing identity metadata.
        Version 1 did not authenticate metadata, so migration cannot prove its
        earlier integrity retrospectively.
        """
        self.db.execute("ALTER TABLE scopes ADD COLUMN erasure_generation INTEGER NOT NULL DEFAULT 0 CHECK(erasure_generation>=0)")
        self.db.execute("CREATE INDEX checkpoint_live_outputs ON checkpoint_requests(principal,namespace,session,revision) WHERE result IS NOT NULL")
        for table in ('checkpoints', 'checkpoint_requests'):
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN identity_digest TEXT NOT NULL DEFAULT ''")
            for row in self.db.execute(f"SELECT * FROM {table}"):
                identity = _checkpoint_identity(row)
                if table == 'checkpoints':
                    envelope = _decode(row['payload'], row['digest'])
                    if envelope['manifest'] != json.loads(row['manifest']):
                        raise IntegrityError('legacy checkpoint manifest differs from sealed state')
                    _decode(row['result'], row['result_digest'])
                elif row['result'] is not None:
                    _decode(row['result'], row['digest'])
                self.db.execute(f"UPDATE {table} SET identity_digest=? WHERE principal=? AND namespace=? AND session=?"
                                + (" AND request=?" if table == 'checkpoint_requests' else ''),
                                (identity, row['principal'], row['namespace'], row['session'])
                                + ((row['request'],) if table == 'checkpoint_requests' else ()))

    def erasure_generation(self) -> int:
        """O(1) scope generation for rejecting work started before an erasure."""
        row = self.db.execute("SELECT erasure_generation FROM scopes WHERE principal=? AND namespace=?", self.scope).fetchone()
        if row is None or type(row[0]) is not int or row[0] < 0:
            raise IntegrityError('invalid scope erasure generation')
        return row[0]

    def _expire_checkpoint_outputs(self, where: str, args: tuple) -> None:
        # Clear and re-seal only live outputs; retained request identities remain
        # verifiable without retaining a digest of erased response text.
        rows = self.db.execute("SELECT * FROM checkpoint_requests WHERE result IS NOT NULL AND "+where, args)
        for row in rows:
            if row['identity_digest'] != _checkpoint_identity(row):
                raise IntegrityError('checkpoint receipt identity checksum mismatch')
            expired = dict(row, result=None, digest='')
            self.db.execute("UPDATE checkpoint_requests SET result=NULL,digest='',identity_digest=? "
                            "WHERE principal=? AND namespace=? AND session=? AND request=?",
                            (_checkpoint_identity(expired),row['principal'],row['namespace'],row['session'],row['request']))

    def _invalidate_continuations(self) -> None:
        self.db.execute("DELETE FROM checkpoints WHERE principal=? AND namespace=?", self.scope)
        self._expire_checkpoint_outputs("principal=? AND namespace=?", self.scope)

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _session(self, session_id: str):
        _identifier(session_id, "session_id")
        row = self.db.execute("SELECT deleted FROM sessions WHERE principal=? AND namespace=? AND session=?",
                              (*self.scope, session_id)).fetchone()
        if row and row[0]:
            raise DeletedError("session was erased; start a new session ID")
        self.db.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?,?,0)",
                        (*self.scope, session_id, _now()))

    def _live_source(self, event_id: str):
        row = self.db.execute("SELECT * FROM messages WHERE principal=? AND namespace=? AND event=?",
                              (*self.scope, event_id)).fetchone()
        if row is None:
            raise ValueError("source event is not present in this memory scope")
        if row["deleted"]:
            raise DeletedError("source event was erased")
        self._message(row)  # checksum before trusting provenance
        return row

    @staticmethod
    def _message(row):
        if _digest(_json([row["session"], row["role"], row["content"]])) != row["digest"]:
            raise IntegrityError("stored source message checksum mismatch")
        return dict(event_id=row["event"], session_id=row["session"], role=row["role"],
                    content=row["content"], seq=row["seq"], created_at=row["created"])

    def append_message(self, session_id: str, role: str, content: str, *, event_id: str,
                       source_event_ids: list[str] | tuple[str, ...] = (),
                       expected_fact_revisions: dict[str, int] | None = None) -> dict:
        _identifier(event_id, "event_id")
        _identifier(session_id, "session_id")
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError("unsupported message role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be nonempty text")
        if len(content.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("message exceeds byte limit")
        digest = _digest(_json([session_id, role, content]))
        if not isinstance(source_event_ids,(list,tuple)) or len(source_event_ids)>128:
            raise ValueError("at most 128 source dependencies are allowed")
        sources=sorted(set(_identifier(x,"source_event_id") for x in source_event_ids))
        if event_id in sources:
            raise ValueError("a message cannot cite itself")
        expected_fact_revisions={} if expected_fact_revisions is None else expected_fact_revisions
        if not isinstance(expected_fact_revisions,dict) or len(expected_fact_revisions)>100:
            raise ValueError("at most 100 fact dependencies are allowed")
        expected_fact_revisions={_identifier(k,"fact key"):_revision(v)
                                 for k,v in expected_fact_revisions.items()}
        with self._transaction():
            self._session(session_id)
            row = self.db.execute("SELECT * FROM messages WHERE principal=? AND namespace=? AND event=?",
                                  (*self.scope, event_id)).fetchone()
            if row:
                if row["deleted"]:
                    raise DeletedError("message was erased; cannot replay its ID")
                if row["digest"] != digest:
                    raise ConflictError("event_id already belongs to different message bytes")
                stored_sources=[r[0] for r in self.db.execute(
                    "SELECT source FROM message_sources WHERE principal=? AND namespace=? AND event=? ORDER BY source",
                    (*self.scope,event_id))]
                if stored_sources!=sources:
                    raise ConflictError("event_id already belongs to different source dependencies")
                return self._message(row)
            for source in sources:
                self._live_source(source)
            for key,revision in expected_fact_revisions.items():
                current=self.db.execute("SELECT revision,deleted FROM facts WHERE principal=? AND namespace=? AND key=?",
                                        (*self.scope,key)).fetchone()
                # Absence is revision zero. An erased row keeps its monotonically
                # increasing revision, so a response based on stable absence can
                # commit while an intervening insertion/recreation still fails.
                observed_revision = current["revision"] if current is not None else 0
                if observed_revision != revision:
                    raise ConflictError("a recalled fact changed before the response committed")
            now = _now()
            cur = self.db.execute("INSERT INTO messages(principal,namespace,event,session,role,content,digest,created) "
                                  "VALUES (?,?,?,?,?,?,?,?)", (*self.scope,event_id,session_id,role,content,digest,now))
            self.db.execute("INSERT INTO message_search(rowid,content) VALUES (?,?)", (cur.lastrowid,content))
            self.db.executemany("INSERT INTO message_sources VALUES (?,?,?,?)",
                                [(*self.scope,event_id,source) for source in sources])
            return dict(event_id=event_id,session_id=session_id,role=role,content=content,
                        seq=cur.lastrowid,created_at=now)

    def get_message(self, event_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM messages WHERE principal=? AND namespace=? AND event=?",
                              (*self.scope,event_id)).fetchone()
        if row and row["deleted"]:
            raise DeletedError("message was erased")
        return self._message(row) if row else None

    def messages(self, session_id: str | None = None, *, limit: int = 50,
                 query: str | None = None, exclude_event_id: str | None = None) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        where = "m.principal=? AND m.namespace=? AND m.deleted=0"
        args: list[Any] = list(self.scope)
        if session_id is not None:
            _identifier(session_id,"session_id")
            where += " AND m.session=?"
            args.append(session_id)
        if exclude_event_id is not None:
            _identifier(exclude_event_id,"exclude_event_id")
            where+=" AND m.event!=?"; args.append(exclude_event_id)
        if query is not None:
            if not isinstance(query,str) or len(query) > 16384:
                raise ValueError("query must be text of at most 16384 characters")
            terms = list(dict.fromkeys(re.findall(r"\w+",query,flags=re.UNICODE)))[:32]
            if not terms:
                return []
            match = " OR ".join('"'+term.replace('"','""')+'"' for term in terms)
            sql = ("SELECT m.* FROM message_search JOIN messages m ON m.seq=message_search.rowid "
                   "WHERE "+where+" AND message_search MATCH ? "
                   "ORDER BY cog_lexical_score(m.content,?) DESC,m.seq DESC LIMIT ?")
            args.extend((match,query,limit))
        else:
            sql = "SELECT m.* FROM messages m WHERE "+where+" ORDER BY m.seq DESC LIMIT ?"
            args.append(limit)
        return [self._message(r) for r in self.db.execute(sql,args)]

    def propose(self, key: str, value: Any, source_event_id: str, *, proposal_id: str,
                expected_revision: int = 0) -> dict:
        _identifier(key,"key"); _identifier(proposal_id,"proposal_id")
        _identifier(source_event_id,"source_event_id"); _revision(expected_revision)
        encoded = _json(value,MAX_MESSAGE_BYTES)
        digest = _digest(_json([key,value,source_event_id,expected_revision]))
        with self._transaction():
            self._live_source(source_event_id)
            row = self.db.execute("SELECT * FROM proposals WHERE principal=? AND namespace=? AND proposal=?",
                                  (*self.scope,proposal_id)).fetchone()
            if row:
                if row["status"] == "erased":
                    raise DeletedError("proposal was erased")
                if row["digest"] != digest:
                    raise ConflictError("proposal ID already used with different content")
                return self._proposal(row)
            now = _now()
            self.db.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (*self.scope,proposal_id,key,encoded,digest,source_event_id,expected_revision,"proposed",now))
            return dict(proposal_id=proposal_id,key=key,value=value,source_event_id=source_event_id,
                        expected_revision=expected_revision,status="proposed",created_at=now)

    @staticmethod
    def _proposal(row):
        try:
            val = json.loads(row["value"])
        except (ValueError,TypeError) as exc:
            raise IntegrityError("invalid proposal bytes") from exc
        if _digest(_json([row["key"],val,row["source"],row["base_revision"]])) != row["digest"]:
            raise IntegrityError("proposal checksum mismatch")
        return dict(proposal_id=row["proposal"],key=row["key"],value=val,
                    source_event_id=row["source"],expected_revision=row["base_revision"],
                    status=row["status"],created_at=row["created"])

    @staticmethod
    def _fact(row):
        val = None if row["deleted"] else _decode(row["value"],row["digest"])
        return dict(key=row["key"],value=val,revision=row["revision"],
                    source_event_id=row["source"],proposal_id=row["proposal"],
                    authority=row["authority"],changed_at=row["changed"],deleted=bool(row["deleted"]))

    def accept(self, proposal_id: str, *, authority: str) -> dict:
        """Trusted host policy approves a proposal; this is not a truth oracle.

        Model confidence is deliberately not an acceptance criterion. ``authority``
        records who/what authorized a write, not a claim of external verification.
        """
        _identifier(proposal_id,"proposal_id"); _identifier(authority,"authority")
        with self._transaction():
            row = self.db.execute("SELECT * FROM proposals WHERE principal=? AND namespace=? AND proposal=?",
                                  (*self.scope,proposal_id)).fetchone()
            if row is None:
                raise ValueError("unknown proposal")
            if row["status"] == "erased":
                raise DeletedError("proposal was erased")
            proposal = self._proposal(row)
            self._live_source(row["source"])
            if row["status"] == "accepted":
                old = self.db.execute("SELECT * FROM fact_history WHERE principal=? AND namespace=? AND proposal=?",
                                      (*self.scope,proposal_id)).fetchone()
                if old is None or old["deleted"]:
                    raise DeletedError("accepted claim was erased")
                if old["authority"] != authority:
                    raise ConflictError("accepted proposal authority cannot be rewritten")
                return self._fact(old)
            current = self.db.execute("SELECT rowid AS search_row,* FROM facts WHERE principal=? AND namespace=? AND key=?",
                                      (*self.scope,row["key"])).fetchone()
            revision = current["revision"] if current else 0
            if revision != row["base_revision"]:
                raise ConflictError(f"fact revision changed: expected {row['base_revision']}, got {revision}")
            if current and not current["deleted"]:
                self._fact(current)
                self.db.execute("DELETE FROM fact_search WHERE rowid=?",(current['search_row'],))
            revision += 1
            record = (*self.scope,row["key"],revision,row["value"],_digest(row["value"]),
                      row["source"],proposal_id,authority,_now(),0)
            fact_cursor=self.db.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",record)
            self.db.execute("INSERT INTO fact_search(rowid,content) VALUES (?,?)",
                            (fact_cursor.lastrowid,row['key']+' '+row['value']))
            self.db.execute("INSERT INTO fact_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",record)
            self.db.execute("UPDATE proposals SET status='accepted' WHERE principal=? AND namespace=? AND proposal=?",
                            (*self.scope,proposal_id))
            return self._fact(self.db.execute("SELECT * FROM facts WHERE principal=? AND namespace=? AND key=?",
                                             (*self.scope,proposal["key"])).fetchone())

    def facts(self, key: str | None = None) -> list[dict]:
        where = "principal=? AND namespace=? AND deleted=0"
        args: tuple = self.scope
        if key is not None:
            _identifier(key,"key"); where += " AND key=?"; args += (key,)
        return [self._fact(r) for r in self.db.execute("SELECT * FROM facts WHERE "+where+" ORDER BY key",args)]

    def fact_revision(self, key: str) -> int:
        row = self.db.execute("SELECT revision FROM facts WHERE principal=? AND namespace=? AND key=?",
                              (*self.scope,_identifier(key,"key"))).fetchone()
        return row[0] if row else 0

    def fact_snapshot(self, key: str) -> dict:
        """Read one fact and its revision from the same SQLite snapshot.

        ``record`` is None for a missing or erased fact. Missing keys have
        revision zero; erased keys retain their deletion revision. Pass this
        revision to ``append_message(expected_fact_revisions=...)`` so an answer
        based on absence cannot commit after another writer inserts the fact.
        """
        key = _identifier(key, "key")
        row = self.db.execute("SELECT * FROM facts WHERE principal=? AND namespace=? AND key=?",
                              (*self.scope, key)).fetchone()
        return dict(key=key, revision=row["revision"] if row is not None else 0,
                    record=self._fact(row) if row is not None and not row["deleted"] else None)

    def history(self, key: str) -> list[dict]:
        return [self._fact(r) for r in self.db.execute(
            "SELECT * FROM fact_history WHERE principal=? AND namespace=? AND key=? ORDER BY revision",
            (*self.scope,_identifier(key,"key")))]

    def recall(self, query: str, *, limit: int = 8, max_chars: int = 8000,
               exclude_event_id: str | None = None) -> dict:
        """Bounded lexical retrieval; no claim of learned semantic relevance."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(max_chars) is not int or not 256 <= max_chars <= 1_000_000:
            raise ValueError("max_chars must be between 256 and 1000000")
        if not isinstance(query,str) or len(query) > 16384:
            raise ValueError("invalid recall query")
        terms = set(re.findall(r"\w+",query.casefold()))
        # Direct key/value matches outrank broad source-related fallback. A source
        # can contain many independent facts; accepting an unrelated one must not
        # displace the fact that actually matches this query.
        observations = self.messages(limit=limit,query=query if query.strip() else None,
                                     exclude_event_id=exclude_event_id)
        source_ids = [m["event_id"] for m in observations]
        facts = []
        if terms:
            match = ' OR '.join('"'+term.replace('"','""')+'"' for term in sorted(terms)[:32])
            sql = ("SELECT * FROM facts WHERE principal=? AND namespace=? AND deleted=0 "
                   "AND rowid IN (SELECT rowid FROM fact_search WHERE fact_search MATCH ?) "
                   "ORDER BY cog_lexical_score(key||' '||value,?) DESC,changed DESC,key LIMIT ?")
            facts = [self._fact(r) for r in self.db.execute(sql,(*self.scope,match,query,limit))]
        if len(facts)<limit and (source_ids or not query.strip()):
            where = "principal=? AND namespace=? AND deleted=0"
            args: list[Any] = list(self.scope)
            if query.strip():
                where += " AND source IN ("+','.join('?' for _ in source_ids)+")"
                args.extend(source_ids)
            if facts:
                where += " AND key NOT IN ("+','.join('?' for _ in facts)+")"
                args.extend(f['key'] for f in facts)
            sql = "SELECT * FROM facts WHERE "+where+" ORDER BY changed DESC,key LIMIT ?"
            facts.extend(self._fact(r) for r in self.db.execute(sql,(*args,limit-len(facts))))
        result = dict(facts=[],observations=[],truncated=False)
        remaining = [("facts",f) for f in facts] + [("observations",m) for m in observations]
        for kind,value in remaining:
            if len(result["facts"])+len(result["observations"]) >= limit:
                result["truncated"] = True
                break
            result[kind].append(value)
            if len(_json(result)) > max_chars:
                result[kind].pop()
                result["truncated"] = True
        return result

    def _forget_single_event(self, event_id: str) -> dict:
        row = self.db.execute("SELECT * FROM messages WHERE principal=? AND namespace=? AND event=?",
                              (*self.scope,event_id)).fetchone()
        if row is None:
            raise ValueError("unknown event in this scope")
        if row["deleted"]:
            return dict(event_id=event_id,erased=True,already_erased=True)
        now = _now()
        self.db.execute("DELETE FROM message_search WHERE rowid=?",(row["seq"],))
        self.db.execute("UPDATE messages SET content=NULL,digest='',deleted=1 WHERE seq=?",(row["seq"],))
        self.db.execute("UPDATE proposals SET value=NULL,digest='',status='erased' "
                        "WHERE principal=? AND namespace=? AND source=?",(*self.scope,event_id))
        current = self.db.execute("SELECT rowid AS search_row,* FROM facts WHERE principal=? AND namespace=? AND source=? AND deleted=0",
                                  (*self.scope,event_id)).fetchall()
        for f in current:
            self.db.execute("DELETE FROM fact_search WHERE rowid=?",(f['search_row'],))
            record = (*self.scope,f["key"],f["revision"]+1,None,"",None,None,"erasure",now,1)
            self.db.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",record)
            self.db.execute("INSERT INTO fact_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",record)
        self.db.execute("UPDATE fact_history SET value=NULL,digest='',deleted=1 "
                        "WHERE principal=? AND namespace=? AND source=?",(*self.scope,event_id))
        self.db.execute("INSERT OR REPLACE INTO erasures VALUES (?,?,?,?)",(*self.scope,event_id,now))
        self.db.execute("UPDATE scopes SET erasure_generation=erasure_generation+1 WHERE principal=? AND namespace=?",self.scope)
        return dict(event_id=event_id,erased=True,already_erased=False)

    def _forget_event(self, event_id: str) -> dict:
        # Dependency edges always point to earlier committed messages. The visited
        # set still makes deletion robust to imported/corrupt cyclic metadata.
        pending=[event_id]; visited=set(); result=None
        while pending:
            current=pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(r[0] for r in self.db.execute(
                "SELECT event FROM message_sources WHERE principal=? AND namespace=? AND source=?",
                (*self.scope,current)))
            erased=self._forget_single_event(current)
            if current==event_id:
                result=erased
        result["derived_messages_erased"]=len(visited)-1
        return result

    def forget_event(self, event_id: str) -> dict:
        _identifier(event_id,"event_id")
        with self._transaction():
            generation = self.erasure_generation()
            result = self._forget_event(event_id)
            if self.erasure_generation() != generation:
                # Decode state may encode any erased source. Invalidate once per
                # operation, including all transitively erased descendants.
                self._invalidate_continuations()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return result

    def forget_session(self, session_id: str) -> dict:
        _identifier(session_id,"session_id")
        with self._transaction():
            prior=self.db.execute("SELECT deleted FROM sessions WHERE principal=? AND namespace=? AND session=?",
                                  (*self.scope,session_id)).fetchone()
            if prior and prior[0]:
                return dict(session_id=session_id,erased=True,messages=0,already_erased=True)
            self._session(session_id)
            events = [r[0] for r in self.db.execute("SELECT event FROM messages WHERE principal=? AND namespace=? AND session=? AND deleted=0",
                                                   (*self.scope,session_id))]
            for event in events:
                self._forget_event(event)
            self._invalidate_continuations()
            self.db.execute("UPDATE sessions SET deleted=1 WHERE principal=? AND namespace=? AND session=?",(*self.scope,session_id))
            # Retiring an empty decoder-only session is also an actual erasure.
            self.db.execute("UPDATE scopes SET erasure_generation=erasure_generation+1 WHERE principal=? AND namespace=?",self.scope)
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return dict(session_id=session_id,erased=True,messages=len(events))

    def load_checkpoint(self, session_id: str) -> dict | None:
        _identifier(session_id,"session_id")
        session = self.db.execute("SELECT deleted FROM sessions WHERE principal=? AND namespace=? AND session=?",
                                  (*self.scope,session_id)).fetchone()
        if session and session[0]:
            raise DeletedError("session was erased")
        r = self.db.execute("SELECT * FROM checkpoints WHERE principal=? AND namespace=? AND session=?",(*self.scope,session_id)).fetchone()
        if r is None:
            # Erasure invalidates state while retaining applied request IDs. A new
            # session ID is necessary, never silently continue from an empty state.
            previous = self.db.execute("SELECT 1 FROM checkpoint_requests WHERE principal=? AND namespace=? AND session=? LIMIT 1",
                                       (*self.scope,session_id)).fetchone()
            if previous:
                raise DeletedError("continuation state invalidated by erasure; start a new session")
            return None
        if r['identity_digest'] != _checkpoint_identity(r):
            raise IntegrityError('checkpoint identity checksum mismatch')
        envelope = _decode(r["payload"],r["digest"])
        manifest = json.loads(r["manifest"])
        if envelope["manifest"] != manifest:
            raise IntegrityError("checkpoint manifest differs from sealed state")
        return dict(revision=r["revision"],manifest=manifest,payload=envelope["payload"],
                    request_id=r["request"],request_hash=r["request_hash"],
                    result=_decode(r["result"],r["result_digest"]))

    def checkpoint_receipt(self, session_id: str, request_id: str) -> dict | None:
        _identifier(session_id,"session_id"); _identifier(request_id,"request_id")
        r = self.db.execute("SELECT * FROM checkpoint_requests WHERE principal=? AND namespace=? AND session=? AND request=?",
                            (*self.scope,session_id,request_id)).fetchone()
        if r is None:
            return None
        if r['identity_digest'] != _checkpoint_identity(r):
            raise IntegrityError('checkpoint receipt identity checksum mismatch')
        result = _decode(r["result"],r["digest"]) if r["result"] is not None else None
        return dict(request_id=request_id,request_hash=r["request_hash"],revision=r["revision"],
                    manifest=json.loads(r["manifest"]),result=result)

    def commit_checkpoint(self, session_id: str, *, request_id: str, request_hash: str,
                          expected_revision: int, manifest: dict, payload: dict, result: dict,
                          expected_erasure_generation: int | None = None) -> dict:
        _identifier(session_id,"session_id"); _identifier(request_id,"request_id")
        _revision(expected_revision)
        if expected_erasure_generation is not None:
            _revision(expected_erasure_generation)
        if not isinstance(request_hash,str) or re.fullmatch(r"[0-9a-f]{64}",request_hash) is None:
            raise ValueError("request_hash must be a SHA-256 hex digest")
        if any(not isinstance(x,dict) for x in (manifest,payload,result)):
            raise ValueError("checkpoint components must be JSON objects")
        encoded = _json(dict(manifest=manifest,payload=payload))
        man = _json(manifest); output = _json(result)
        with self._transaction():
            self._session(session_id)
            if (expected_erasure_generation is not None
                    and self.erasure_generation() != expected_erasure_generation):
                raise DeletedError('memory was erased while continuation was computing')
            old = self.checkpoint_receipt(session_id,request_id)
            if old:
                if old["request_hash"] != request_hash or old["manifest"] != manifest:
                    raise ConflictError("checkpoint request ID conflicts with earlier input/model")
                return dict(old,duplicate=True)
            current = self.load_checkpoint(session_id)
            revision = current["revision"] if current else 0
            if revision != expected_revision:
                raise ConflictError(f"checkpoint revision changed: expected {expected_revision}, got {revision}")
            if current and current["manifest"] != manifest:
                raise ConflictError("checkpoint model/schema identity changed")
            revision += 1
            checkpoint = dict(principal=self.principal_id,namespace=self.namespace,session=session_id,
                              request=request_id,request_hash=request_hash,revision=revision,manifest=man,
                              payload=encoded,digest=_digest(encoded),result_digest=_digest(output))
            receipt = dict(checkpoint)
            del receipt['payload']
            receipt['digest'] = _digest(output)
            self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (*self.scope,session_id,revision,man,encoded,_digest(encoded),request_id,request_hash,output,_digest(output),_checkpoint_identity(checkpoint)))
            self.db.execute("INSERT INTO checkpoint_requests VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (*self.scope,session_id,request_id,request_hash,revision,man,output,_digest(output),_checkpoint_identity(receipt)))
            self._expire_checkpoint_outputs("principal=? AND namespace=? AND session=? AND revision<=?",
                                            (*self.scope,session_id,revision-CHECKPOINT_OUTPUTS))
            # Also bound retained output bytes. Keep the newest receipt even if a
            # caller explicitly supplied one result larger than the cache budget.
            outputs=self.db.execute("SELECT request,revision,length(CAST(result AS BLOB)) AS size "
                                    "FROM checkpoint_requests WHERE principal=? AND namespace=? AND session=? "
                                    "AND result IS NOT NULL ORDER BY revision DESC",(*self.scope,session_id)).fetchall()
            retained=0
            for cached in outputs:
                retained+=cached['size']
                if retained>CHECKPOINT_OUTPUT_BYTES and cached['revision']!=revision:
                    self._expire_checkpoint_outputs("principal=? AND namespace=? AND session=? AND request=?",
                                                    (*self.scope,session_id,cached['request']))
            return dict(request_id=request_id,request_hash=request_hash,revision=revision,
                        manifest=manifest,result=result,duplicate=False)

    def export_json(self, destination: str | Path) -> dict:
        """Export this scope's readable history, assertions and deletion markers."""
        path=Path(destination).expanduser().absolute()
        if path.exists():
            raise FileExistsError("export destination must be new")
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd,name=tempfile.mkstemp(prefix=".cog-export-",dir=path.parent); temp=Path(name)
        digest=hashlib.sha256(); count=0
        try:
            with os.fdopen(fd,"wb") as out:
                def write(text):
                    nonlocal count
                    data=text.encode("utf-8"); out.write(data); digest.update(data); count+=len(data)
                # Stream rows under one read snapshot; memory cost does not grow
                # with the entire lifetime archive. The result remains ordinary JSON.
                self.db.execute("BEGIN")
                try:
                    header=dict(format="nanocog-memory-export-v1",principal_id=self.principal_id,
                                namespace=self.namespace,created_at=_now())
                    write(_json(header)[:-1]+',"tables":{')
                    names=("sessions","messages","message_sources","proposals","facts","fact_history","erasures")
                    for i,table in enumerate(names):
                        write((',' if i else '')+_json(table)+':[')
                        rows=self.db.execute(f"SELECT * FROM {table} WHERE principal=? AND namespace=?",self.scope)
                        for j,row in enumerate(rows):
                            write((',' if j else '')+_json(dict(row)))
                        write(']')
                    write('}}')
                finally:
                    self.db.execute("ROLLBACK")
                out.flush(); os.fsync(out.fileno())
            _publish_file(temp,path)
        finally:
            temp.unlink(missing_ok=True)
        return dict(path=str(path),sha256=digest.hexdigest(),bytes=count)

    def backup(self, destination: str | Path) -> dict:
        """Consistent full-database local backup, including every principal.

        This administrative operation is for the local database owner. Copy the
        database and its .manifest.json together to independently durable storage.
        """
        path = Path(destination).expanduser().absolute()
        manifest_path = Path(str(path)+".manifest.json")
        _unused_database_path(path)
        if os.path.lexists(manifest_path) or path == self.path:
            raise FileExistsError("backup destination must be new")
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd,name = tempfile.mkstemp(prefix=".cog-backup-",dir=path.parent)
        os.close(fd); temp=Path(name)
        try:
            with closing(sqlite3.connect(temp)) as target:
                self.db.backup(target)
                target.execute("PRAGMA journal_mode=DELETE")
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise IntegrityError("backup failed SQLite integrity check")
            with temp.open("rb") as stream:
                digest=hashlib.file_digest(stream,"sha256").hexdigest()
            metadata=dict(format="nanocog-memory-backup-v1",sha256=digest,
                          schema_version=SCHEMA_VERSION,created_at=_now(),bytes=temp.stat().st_size)
            _publish_file(temp,path)
            _write_new(manifest_path,(_json(metadata)+"\n").encode("utf-8"))
            return dict(path=str(path),manifest=str(manifest_path),**metadata)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def restore_backup(source: str | Path, destination: str | Path) -> dict:
        """Restore verified backup to a NEW path. Never replace a running database.

        Older backups retain previously erased data. Restore the latest post-erasure
        backup or reapply its erasure list before exposing an older restored copy.
        """
        src=Path(source).expanduser().absolute(); dest=Path(destination).expanduser().absolute()
        _unused_database_path(dest)
        metadata=json.loads(Path(str(src)+".manifest.json").read_text())
        with src.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        if (metadata.get("format") != "nanocog-memory-backup-v1" or metadata.get("sha256") != digest
                or metadata.get("schema_version") not in {1, SCHEMA_VERSION} or metadata.get("bytes") != src.stat().st_size):
            raise IntegrityError("backup identity/bytes do not match manifest")
        # Copy once, validate the actual bytes being installed to avoid a source
        # changing between validation and use. No SQLite extensions or pickle load.
        dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd,name=tempfile.mkstemp(prefix=".cog-restore-",dir=dest.parent); temp=Path(name)
        try:
            with os.fdopen(fd,"wb") as out, src.open("rb") as inp:
                while block:=inp.read(1024*1024): out.write(block)
                out.flush(); os.fsync(out.fileno())
            with temp.open("rb") as f:
                if hashlib.file_digest(f,"sha256").hexdigest()!=digest:
                    raise IntegrityError("backup changed while restoring")
            with closing(sqlite3.connect(temp)) as check:
                if (check.execute("PRAGMA application_id").fetchone()[0]!=APPLICATION_ID or
                    check.execute("PRAGMA user_version").fetchone()[0]!=metadata['schema_version'] or
                    check.execute("PRAGMA integrity_check").fetchone()[0]!="ok"):
                    raise IntegrityError("backup schema or SQLite integrity invalid")
            _publish_file(temp,dest)
            return dict(path=str(dest),sha256=digest,schema_version=metadata['schema_version'])
        finally:
            temp.unlink(missing_ok=True)
