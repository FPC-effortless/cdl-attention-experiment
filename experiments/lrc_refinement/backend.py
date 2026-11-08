"""Frozen-descriptor refinement primitives; no model or target information.

Addresses are ``(coarse_code, fine_prefix)``.  A fine prefix consists of the
first ``depth`` fixed threshold comparisons, packed into the least-significant
bits.  Probe policies see only the reader's coarse ordering, projected query,
and frozen split thresholds.  In particular they cannot inspect occupancy.

The index is immutable and stores every occurrence.  An overflowing bucket
returns no occurrences.  Lookup never scans the memory, truncates a bucket, or
selects a winner by occurrence number.  The caller owns content scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


Address = tuple[int, int]
MAX_DEPTH = 64
MAX_PROBES = 2
MAX_CAPACITY = 2


def _integer(value: object, name: str, *, maximum: int | None = None) -> int:
    """Validate integer scalars without accepting booleans or truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if getattr(value, "ndim", 0) != 0:
        raise TypeError(f"{name} must be an integer scalar")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if result < 0 or (maximum is not None and result > maximum):
        limit = f" through {maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be 0{limit}")
    return int(result)


def prefix_bits(projections, thresholds, depth: int):
    """Pack ``projections[..., :depth] >= thresholds[..., :depth]`` into bits.

    Leading dimensions may broadcast; the final projection dimension must
    match.  Input comparisons use float64.  Return a uint64 array preserving
    the broadcast leading dimensions, or a numpy.uint64 scalar for a vector.
    Depth zero returns zero.  Bit ``i`` represents split ``i`` so refining from
    depth d to a larger depth never changes the first d bits.
    """
    depth = _integer(depth, "depth", maximum=MAX_DEPTH)
    projected = np.asarray(projections, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if projected.ndim < 1 or cuts.ndim < 1:
        raise ValueError("projections and thresholds must have a split dimension")
    if projected.shape[-1] != cuts.shape[-1]:
        raise ValueError("projection and threshold split dimensions differ")
    if depth > projected.shape[-1]:
        raise ValueError("depth exceeds available split descriptors")
    if not np.isfinite(projected).all() or not np.isfinite(cuts).all():
        raise ValueError("split projections and thresholds must be finite")
    projected, cuts = np.broadcast_arrays(projected, cuts)
    bits = (projected[..., :depth] >= cuts[..., :depth]).astype(np.uint64)
    places = np.left_shift(np.uint64(1), np.arange(depth, dtype=np.uint64))
    packed = np.sum(bits * places, axis=-1, dtype=np.uint64)
    return packed[()] if np.ndim(packed) == 0 else packed


def choose_probes(
    coarse_order: Iterable[int],
    projected_query,
    thresholds,
    depth: int,
    policy: str = "two_coarse",
) -> tuple[Address, ...]:
    """Choose at most two addresses without any access to the memory/index.

    ``coarse_order`` contains the reader's first one or two coarse codes in
    preference order; duplicate codes are removed, preserving that order.
    ``thresholds`` has shape (16, maxdepth), and ``projected_query`` has shape
    (maxdepth,).  Both are interpreted as float64.

    ``two_coarse`` probes the matching fine prefix in each supplied coarse
    code.  ``sibling`` probes only the first coarse code: its matching prefix
    followed by the prefix obtained by flipping the smallest absolute-margin
    bit.  Margin ties use the lowest split index, independent of record order.
    At depth zero, ``sibling`` has only one distinct address.
    """
    if policy not in ("two_coarse", "sibling"):
        raise ValueError("unknown probe policy")
    depth = _integer(depth, "depth", maximum=MAX_DEPTH)
    query = np.asarray(projected_query, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if query.ndim != 1 or cuts.ndim != 2 or cuts.shape != (16, query.shape[0]):
        raise ValueError("expected query (maxdepth,) and thresholds (16,maxdepth)")
    if depth > query.shape[0]:
        raise ValueError("depth exceeds available split descriptors")
    if not np.isfinite(query).all() or not np.isfinite(cuts).all():
        raise ValueError("query projections and thresholds must be finite")
    order = tuple(_integer(code, "coarse code", maximum=15) for code in coarse_order)
    if not 1 <= len(order) <= MAX_PROBES:
        raise ValueError("supply only the first one or two coarse codes")
    order = tuple(dict.fromkeys(order))
    if policy == "two_coarse":
        return tuple((code, int(prefix_bits(query, cuts[code], depth))) for code in order)
    code = order[0]
    matching = int(prefix_bits(query, cuts[code], depth))
    if depth == 0:
        return ((code, matching),)
    bit = int(np.argmin(np.abs(query[:depth] - cuts[code, :depth])))
    return ((code, matching), (code, matching ^ (1 << bit)))


@dataclass(frozen=True, slots=True)
class LookupResult:
    code: int
    prefix: int
    candidates: tuple[int, ...]
    bucket_size: int
    overflow: bool

    @property
    def empty(self) -> bool:
        """An absent address; an overflowing address is not empty."""
        return self.bucket_size == 0


@dataclass(frozen=True, slots=True)
class GatherResult:
    candidates: tuple[int, ...]
    probes: int
    empty: int
    overflow: int
    bucketrefs: int
    lookups: tuple[LookupResult, ...]

    @property
    def returned(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RefinedIndex:
    """Immutable address table with explicit fail-closed capacity.

    Construction is O(N).  Each lookup performs one expected O(1) dictionary
    access and reads the bucket length, returning at most ``capacity`` opaque
    occurrence positions.  Capacity may be 0, 1, or 2.  ``gather`` accepts no
    more than two requested probes, deduplicates identical addresses before
    lookup, and deduplicates returned occurrence positions.  It never skips
    empty/overflowing addresses in favor of an unrequested populated address.

    ``bucketrefs`` is the sum of the complete bucket *lengths*, including
    overflowed buckets.  It is metadata, not a count of candidate scores or
    payload reads.  Candidate scores are charged separately by the caller.
    ``buckets`` is an explicit read-only audit view, not a query search API.
    """

    _buckets: Mapping[Address, tuple[int, ...]]
    capacity: int
    stored_records: int

    def __init__(self, codes: Iterable[int], prefixes: Iterable[int], capacity: int = 2):
        capacity = _integer(capacity, "capacity", maximum=MAX_CAPACITY)
        if getattr(codes, "ndim", 1) != 1 or getattr(prefixes, "ndim", 1) != 1:
            raise ValueError("codes and prefixes must be one-dimensional")
        codes = tuple(_integer(code, "code") for code in codes)
        prefixes = tuple(_integer(prefix, "prefix") for prefix in prefixes)
        if len(codes) != len(prefixes):
            raise ValueError("codes and prefixes must have the same length")
        buckets: dict[Address, list[int]] = {}
        for occurrence, address in enumerate(zip(codes, prefixes)):
            buckets.setdefault(address, []).append(occurrence)
        immutable = {address: tuple(positions) for address, positions in buckets.items()}
        object.__setattr__(self, "_buckets", MappingProxyType(immutable))
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "stored_records", len(codes))

    def __len__(self) -> int:
        return self.stored_records

    @property
    def buckets(self) -> Mapping[Address, tuple[int, ...]]:
        return self._buckets

    def lookup(self, code: int, prefix: int) -> LookupResult:
        code = _integer(code, "code")
        prefix = _integer(prefix, "prefix")
        bucket = self._buckets.get((code, prefix), ())
        size = len(bucket)
        overflow = size > self.capacity
        return LookupResult(code, prefix, () if overflow else bucket, size, overflow)

    def gather(self, probes: Iterable[Address]) -> GatherResult:
        requested = []
        for probe in probes:
            if len(requested) == MAX_PROBES:
                raise ValueError("the total requested probe budget is two")
            try:
                code, prefix = probe
            except (TypeError, ValueError) as exc:
                raise ValueError("each probe must be a (code,prefix) pair") from exc
            requested.append((_integer(code, "code"), _integer(prefix, "prefix")))
        unique = tuple(dict.fromkeys(requested))
        lookups = tuple(self.lookup(code, prefix) for code, prefix in unique)
        candidates = tuple(dict.fromkeys(position for hit in lookups for position in hit.candidates))
        if len(candidates) > MAX_PROBES * MAX_CAPACITY:
            raise AssertionError("candidate budget violated")
        return GatherResult(
            candidates=candidates,
            probes=len(lookups),
            empty=sum(hit.empty for hit in lookups),
            overflow=sum(hit.overflow for hit in lookups),
            bucketrefs=sum(hit.bucket_size for hit in lookups),
            lookups=lookups,
        )
