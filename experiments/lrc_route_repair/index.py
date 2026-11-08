"""Bounded lookup of occurrence positions by learned discrete address.

Build once from a one-dimensional row of nonnegative integer write codes. The
model keeps its continuous keys and values separately, aligned to the positions
returned here. An ordinary query performs one dictionary lookup and reads the
bucket length; it neither scans the history nor examines candidate contents.

Every occurrence is retained for audit. A bucket with more than ``max_candidates``
occurrences produces an explicit overflow miss, never a positional prefix. When
a bucket fits, *all* its positions are returned. Thus the candidate set commutes
with a permutation of the input row, after mapping positions through that
permutation. Content scoring, including handling score ties, belongs to the
resolver and is not performed by this index.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import Iterable, Mapping


def _nonnegative_integer(value: object, name: str) -> int:
    """Accept integer scalars without importing a tensor/array framework."""
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim != 0:
        raise TypeError(f"{name} must be an integer scalar")
    dtype = getattr(value, "dtype", None)
    if (
        isinstance(value, bool)
        or getattr(dtype, "kind", None) == "b"
        or str(dtype) == "torch.bool"
    ):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(result)


@dataclass(frozen=True, slots=True)
class LookupResult:
    """A single probe, including explicit empty and overflow outcomes.

    ``candidate_reads`` counts returned occurrence references, not tensor values
    read or neural scores evaluated. A resolver should separately count its own
    content reads. ``dropped_count`` records candidates withheld by the cap;
    overflow withholds the entire bucket and must be scored as a lookup miss.
    """

    code: int
    candidates: tuple[int, ...]
    bucket_size: int
    overflow: bool
    max_candidates: int
    probes: int = 1

    @property
    def candidate_reads(self) -> int:
        return len(self.candidates)

    @property
    def dropped_count(self) -> int:
        return self.bucket_size if self.overflow else 0

    @property
    def empty(self) -> bool:
        """Whether the address has no stored occurrences (not an overflow)."""
        return self.bucket_size == 0

    @property
    def miss(self) -> bool:
        return not self.candidates


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Snapshot of build size and cumulative query work since the last reset."""

    stored_records: int
    occupied_buckets: int
    max_candidates: int
    lookups: int
    candidate_reads: int
    empty_lookups: int
    overflow_lookups: int
    dropped_candidates: int
    max_returned: int

    @property
    def probes(self) -> int:
        return self.lookups


class BoundedKeyIndex:
    """A per-row immutable address table with mutable query counters.

    ``codes`` may be a one-dimensional integer tensor/array or an iterable of
    integer scalars. Tensor/array conversion occurs only during construction.
    Code 0 is a regular address. The nonnegative cap may be 0, in which case
    every nonempty bucket overflows. The index assumes no labels, entity names,
    sentence grammar, or temporal ordering.

    Construction is O(N) time and storage. Each lookup uses a single expected
    O(1) dictionary access, and returns at most K occurrence references, where
    K is the cap. The resolver's bounded gather/scoring cost is separate. Audit
    accessors are intentionally explicit and are not called by ``lookup``.
    """

    def __init__(self, codes: Iterable[int], *, max_candidates: int) -> None:
        self._max_candidates = _nonnegative_integer(max_candidates, "max_candidates")
        ndim = getattr(codes, "ndim", None)
        if ndim is not None:
            if ndim != 1:
                raise ValueError("codes must be one-dimensional")
            tolist = getattr(codes, "tolist", None)
            if tolist is not None:
                codes = tolist()
        try:
            code_iter = iter(codes)
        except TypeError as exc:
            raise TypeError("codes must be a one-dimensional integer iterable") from exc

        buckets: dict[int, list[int]] = {}
        record_count = 0
        for position, value in enumerate(code_iter):
            code = _nonnegative_integer(value, f"codes[{position}]")
            buckets.setdefault(code, []).append(position)
            record_count += 1
        self._buckets = {code: tuple(positions) for code, positions in buckets.items()}
        self._record_count = record_count
        self.reset_stats()

    def __len__(self) -> int:
        return self._record_count

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    def lookup(self, code: int) -> LookupResult:
        """Probe one code without scanning, sorting, or truncating its bucket."""
        query_code = _nonnegative_integer(code, "code")
        bucket = self._buckets.get(query_code, ())
        size = len(bucket)
        overflow = size > self._max_candidates
        candidates = () if overflow else bucket
        reads = 0 if overflow else size

        self._lookups += 1
        self._candidate_reads += reads
        self._empty_lookups += int(size == 0)
        self._overflow_lookups += int(overflow)
        self._dropped_candidates += size if overflow else 0
        self._max_returned = max(self._max_returned, reads)
        return LookupResult(
            code=query_code,
            candidates=candidates,
            bucket_size=size,
            overflow=overflow,
            max_candidates=self._max_candidates,
        )

    @property
    def stats(self) -> IndexStats:
        return IndexStats(
            stored_records=self._record_count,
            occupied_buckets=len(self._buckets),
            max_candidates=self._max_candidates,
            lookups=self._lookups,
            candidate_reads=self._candidate_reads,
            empty_lookups=self._empty_lookups,
            overflow_lookups=self._overflow_lookups,
            dropped_candidates=self._dropped_candidates,
            max_returned=self._max_returned,
        )

    def reset_stats(self) -> None:
        """Reset query counters, leaving stored records unchanged."""
        self._lookups = 0
        self._candidate_reads = 0
        self._empty_lookups = 0
        self._overflow_lookups = 0
        self._dropped_candidates = 0
        self._max_returned = 0

    def audit_bucket(self, code: int) -> tuple[int, ...]:
        """Return a complete bucket, even if overflowing; not an inference API."""
        return self._buckets.get(_nonnegative_integer(code, "code"), ())

    def audit_buckets(self) -> Mapping[int, tuple[int, ...]]:
        """Expose every occurrence through a read-only view for explicit audit."""
        return MappingProxyType(self._buckets)
