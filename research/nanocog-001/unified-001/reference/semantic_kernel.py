"""COG-UNIFIED-001 semantic reference, Python standard library only.

This is not a neural language model. Learned boundaries/keys/values are inputs
to this interface; typed fixtures are oracle/engineering checks only. Storage
is an immutable tuple with linear lookup, not a claimed optimized index.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

Key = tuple[str, str, str]
MAX_REVISION = (1 << 64) - 1
MAX_LINKS = 64
MAX_TRANSACTION_WRITES = 16
MAX_DECLARED_READS = 256


def valid_key(key: Key) -> bool:
    return (isinstance(key, tuple) and len(key) == 3
            and all(isinstance(s, str) and 0 < len(s.encode()) <= 512 for s in key))


@dataclass(frozen=True)
class Epochs:
    stream: str = "stream-0"
    authority: int = 0
    policy: str = "policy-0"
    schema: str = "typed-string-v1"
    model: str = "model-0"
    tokenizer: str = "tokenizer-0"

    def valid(self) -> bool:
        return (type(self.authority) is int and 0 <= self.authority <= MAX_REVISION
                and all(isinstance(x, str) and 0 < len(x.encode()) <= 256
                        for x in (self.stream, self.policy, self.schema,
                                  self.model, self.tokenizer)))


@dataclass(frozen=True)
class Payload:
    kind: str
    value: Optional[str] = None
    target: Optional[Key] = None
    evidence: tuple[str, ...] = ()
    producer: str = "reference"

    def valid(self) -> bool:
        metadata = (isinstance(self.evidence, tuple) and len(self.evidence) <= 8
                    and all(isinstance(x, str) and len(x.encode()) <= 256
                            for x in self.evidence)
                    and isinstance(self.producer, str)
                    and len(self.producer.encode()) <= 256)
        if not metadata:
            return False
        if self.kind == "value":
            return (isinstance(self.value, str) and len(self.value.encode()) <= 4096
                    and self.target is None)
        if self.kind == "link":
            return self.value is None and valid_key(self.target)
        if self.kind == "tombstone":
            return self.value is None and self.target is None
        return False


@dataclass(frozen=True)
class Cell:
    revision: int
    payload: Payload
    schema: str

    def valid(self) -> bool:
        return (type(self.revision) is int and 1 <= self.revision <= MAX_REVISION
                and isinstance(self.payload, Payload) and self.payload.valid()
                and isinstance(self.schema, str) and 0 < len(self.schema.encode()) <= 256)


@dataclass(frozen=True)
class Store:
    entries: tuple[tuple[Key, Cell], ...] = ()
    capacity: int = 1024
    epochs: Epochs = Epochs()

    def __post_init__(self):
        if type(self.capacity) is not int or not 0 <= self.capacity <= (1 << 31) - 1:
            raise ValueError("capacity must be a nonnegative integer")
        if not isinstance(self.epochs, Epochs) or not self.epochs.valid():
            raise ValueError("invalid epoch metadata")
        if not isinstance(self.entries, tuple) or len(self.entries) > self.capacity:
            raise ValueError("immutable entries exceed capacity")
        seen = set()
        for entry in self.entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("entries must be immutable key-cell pairs")
            key, cell = entry
            if not valid_key(key) or key in seen:
                raise ValueError("invalid or duplicate full key")
            seen.add(key)
            if not isinstance(cell, Cell) or not cell.valid():
                raise ValueError("invalid cell")

    def cell(self, key: Key) -> Optional[Cell]:
        for k, cell in self.entries:
            if k == key:
                return cell
        return None

    def revision(self, key: Key) -> int:
        cell = self.cell(key)
        return 0 if cell is None else cell.revision


@dataclass(frozen=True)
class Resolution:
    status: str
    value: Optional[str]
    # Full immutable observations also disambiguate forks with equal counters.
    dependencies: tuple[tuple[Key, Optional[Cell]], ...]


def resolve(store: Store, key: Key, max_links: int = 4) -> Resolution:
    """Pure read. A cap of H permits H link traversals plus the terminal read."""
    if (not valid_key(key) or type(max_links) is not int
            or not 0 <= max_links <= MAX_LINKS):
        raise ValueError("invalid key or link cap")
    seen: set[Key] = set()
    dependencies: list[tuple[Key, Optional[Cell]]] = []
    traversed = 0
    current = key
    while True:
        if current in seen:
            return Resolution("cycle", None, tuple(dependencies))
        seen.add(current)
        cell = store.cell(current)
        dependencies.append((current, cell))
        if cell is None:
            return Resolution("missing", None, tuple(dependencies))
        if cell.schema != store.epochs.schema:
            return Resolution("incompatible", None, tuple(dependencies))
        payload = cell.payload
        if payload.kind == "tombstone":
            return Resolution("deleted", None, tuple(dependencies))
        if payload.kind == "value":
            return Resolution("ok", payload.value, tuple(dependencies))
        if traversed >= max_links:
            return Resolution("hop_limit", None, tuple(dependencies))
        current = payload.target
        traversed += 1


@dataclass(frozen=True)
class Operation:
    kind: str
    destination: Key
    value: Optional[str] = None
    source: Optional[Key] = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transaction:
    epochs: Epochs
    expected: tuple[tuple[Key, Optional[Cell]], ...]
    writes: tuple[tuple[Key, Payload], ...]


def prepare(store: Store, operations: Iterable[Operation], *,
            observed: Iterable[Key] = (), max_links: int = 4) -> Transaction:
    """Prepare on one snapshot. Caller must disclose predicate read dependencies.

    COPY reads all dereferenced cells itself. All COPY operations in a batch
    read the original snapshot, not earlier writes in the same batch.
    """
    if type(max_links) is not int or not 0 <= max_links <= MAX_LINKS:
        raise ValueError("invalid link cap")
    expected: dict[Key, Optional[Cell]] = {}
    writes: dict[Key, Payload] = {}
    for count, key in enumerate(observed):
        if count >= MAX_DECLARED_READS:
            raise ValueError("declared dependency budget exhausted")
        if not valid_key(key):
            raise ValueError("invalid declared dependency")
        expected[key] = store.cell(key)
    for count, op in enumerate(operations):
        if count >= MAX_TRANSACTION_WRITES:
            raise ValueError("transaction write budget exhausted")
        if not isinstance(op, Operation) or not valid_key(op.destination):
            raise ValueError("invalid operation")
        key = op.destination
        if key in writes:
            raise ValueError("duplicate destination in one transaction")
        expected[key] = store.cell(key)
        if op.kind == "set" and op.source is None:
            payload = Payload("value", value=op.value, evidence=op.evidence,
                              producer=store.epochs.model)
        elif op.kind == "delete" and op.value is None and op.source is None:
            payload = Payload("tombstone", evidence=op.evidence,
                              producer=store.epochs.model)
        elif op.kind == "link" and op.value is None and valid_key(op.source):
            payload = Payload("link", target=op.source, evidence=op.evidence,
                              producer=store.epochs.model)
        elif op.kind == "copy" and op.value is None and valid_key(op.source):
            read = resolve(store, op.source, max_links)
            if read.status != "ok":
                raise ValueError("COPY source is unresolved: " + read.status)
            expected.update(read.dependencies)
            payload = Payload("value", value=read.value, evidence=op.evidence,
                              producer=store.epochs.model)
        else:
            raise ValueError("unknown or malformed operation")
        if not payload.valid():
            raise ValueError("invalid payload")
        writes[key] = payload
    return Transaction(store.epochs, tuple(sorted(expected.items())),
                       tuple(sorted(writes.items())))


@dataclass(frozen=True)
class Commit:
    accepted: bool
    status: str
    store: Store


def commit(store: Store, tx: Transaction, *, authorized: bool = True,
           protected: frozenset[Key] = frozenset()) -> Commit:
    """All writes or none in this single-process immutable reference.

    `authorized` is a supplied policy verdict, not semantic truth or proof of
    external authorization. Epoch checks do not make that verdict sound.
    """
    def reject(reason: str) -> Commit:
        return Commit(False, reason, store)

    if authorized is not True:
        return reject("not_authorized")
    if (not isinstance(tx, Transaction) or not isinstance(tx.epochs, Epochs)
            or not tx.epochs.valid()):
        return reject("malformed")
    if tx.epochs != store.epochs:
        return reject("epoch_conflict")
    if not isinstance(tx.expected, tuple) or not isinstance(tx.writes, tuple):
        return reject("malformed")
    expected: dict[Key, Optional[Cell]] = {}
    if len(tx.expected) > MAX_DECLARED_READS + MAX_TRANSACTION_WRITES * (MAX_LINKS + 2):
        return reject("dependency_budget_exhausted")
    if len(tx.writes) > MAX_TRANSACTION_WRITES:
        return reject("write_budget_exhausted")
    for entry in tx.expected:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return reject("malformed")
        key, observation = entry
        if (not valid_key(key) or key in expected
                or (observation is not None and
                    (not isinstance(observation, Cell) or not observation.valid()))):
            return reject("malformed")
        expected[key] = observation
    writes: dict[Key, Payload] = {}
    for entry in tx.writes:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return reject("malformed")
        key, payload = entry
        if (not valid_key(key) or key in writes or key not in expected
                or not isinstance(payload, Payload) or not payload.valid()):
            return reject("malformed")
        if key in protected:
            return reject("protected_write")
        if store.revision(key) == MAX_REVISION:
            return reject("revision_exhausted")
        writes[key] = payload
    if any(store.cell(key) != observation for key, observation in expected.items()):
        return reject("version_conflict")
    new_keys = sum(store.cell(key) is None for key in writes)
    if len(store.entries) + new_keys > store.capacity:
        return reject("capacity_exhausted")
    updated = dict(store.entries)
    for key, payload in writes.items():
        updated[key] = Cell(store.revision(key) + 1, payload, store.epochs.schema)
    return Commit(True, "committed", Store(tuple(sorted(updated.items())),
                                          store.capacity, store.epochs))


@dataclass(frozen=True)
class ReadCache:
    key: Key
    max_links: int
    epochs: Epochs
    result: Resolution


def cached_resolution(store: Store, key: Key, max_links: int = 4,
                      cache: Optional[ReadCache] = None
                      ) -> tuple[Resolution, ReadCache, bool]:
    """Reuse pure typed reads only; never cache neural output from h/packets.

    A cache is an internal result created by this function, not an authenticated
    certificate accepted from an untrusted caller.
    """
    if (not valid_key(key) or type(max_links) is not int
            or not 0 <= max_links <= MAX_LINKS):
        raise ValueError("invalid key or link cap")
    if (cache is not None and cache.key == key and cache.max_links == max_links
            and cache.epochs == store.epochs
            and all(store.cell(k) == observation for k, observation in cache.result.dependencies)):
        return cache.result, cache, True
    result = resolve(store, key, max_links)
    return result, ReadCache(key, max_links, store.epochs, result), False


@dataclass(frozen=True)
class KernelStep:
    read: Optional[Resolution]
    output: str
    transition: Commit


def semantic_step(store: Store, predicted_key: Optional[Key],
                  proposals: Iterable[Operation],
                  decode: Callable[[Optional[Resolution]], str], *,
                  max_links: int = 4, authorized: bool = True,
                  protected: frozenset[Key] = frozenset()) -> KernelStep:
    """Read/decode old state, then commit the current proposal.

    Inputs stand in for learned interfaces; their truth is not guaranteed.
    Local neural state is intentionally absent from this semantic reference.
    """
    read = None if predicted_key is None else resolve(store, predicted_key, max_links)
    output = decode(read)
    try:
        tx = prepare(store, proposals, max_links=max_links)
    except ValueError:
        transition = Commit(False, "invalid_proposal", store)
    else:
        transition = commit(store, tx, authorized=authorized, protected=protected)
    return KernelStep(read, output, transition)


def render_exact(read: Optional[Resolution]) -> str:
    if read is None:
        return "NO_QUERY"
    return read.value if read.status == "ok" else "UNRESOLVED:" + read.status


def affine_compose(outer: tuple[int, int], inner: tuple[int, int]) -> tuple[int, int]:
    """Exact integer reference. Floating-point reassociation is a separate gate."""
    a, b = outer
    c, d = inner
    return a * c, a * d + b


def affine_apply(transform: tuple[int, int], state: int) -> int:
    a, b = transform
    return a * state + b
