"""Exact rank-paged retrieval over append-only, strictly causal posting lists.

Motivation: a cap C on the most recent occurrences can make *every* older
occurrence unreachable. Here a B-bit page route selects one of the chronological
C-sized pages. If 2**B >= ceil(m/C), every earlier occurrence has a page/slot
address. This is an expressivity and accessibility guarantee, NOT a guarantee
that a neural policy will learn the correct page.

The index is O(1) amortized append and O(log m+C) historical lookup, O(m)
storage, with no dense N*N similarity matrix. At step t immediately after
appending a writer at t, current-step lookups can skip binary search.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PagedRead:
    candidates: tuple[int, ...]
    omitted: int
    pages: int
    chosen_page: int | None
    total_eligible: int


def rank_address(rank: int, cap: int) -> tuple[int, int]:
    """Absolute reverse-chronological rank -> (page, slot)."""
    if rank < 0 or cap < 1:
        raise ValueError('rank must be nonnegative and cap positive')
    return divmod(rank, cap)


def rank_bits(page: int, width: int) -> tuple[int, ...]:
    """Little-endian finite-width binary page code."""
    if width < 1 or page < 0 or page >= (1 << width):
        raise ValueError('page out of addressable range')
    return tuple((page >> i) & 1 for i in range(width))


def bits_page(bits: Sequence[int]) -> int:
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError('expected nonempty sequence of binary digits')
    if len(bits) > 62:
        raise ValueError('bit width exceeds signed 64-bit safe range')
    return sum(int(bit) << k for k, bit in enumerate(bits))


class PagedCausalIndex:
    """Unbounded history, bounded reads, arbitrary historic page accessibility.

    `lookup_page` defaults to page 0 (legacy recency). Page indices beyond
    existing pages return NULL, rather than wrap. A learned router can choose
    a valid page using a masked bit-by-bit policy; modulo is deliberately NOT
    used, since that gives aliasing and biased action probabilities.
    """
    def __init__(self, cap: int):
        if cap < 1:
            raise ValueError('cap must be positive')
        self.cap = cap
        self.postings: dict[tuple[int, int], list[int]] = {}
        self.last_position = -1

    def append(self, position: int, relation: int, address: int):
        if position <= self.last_position or min(position, relation, address) < 0:
            raise ValueError('writes must be nonnegative and strictly ordered')
        self.postings.setdefault((relation, address), []).append(position)
        self.last_position = position

    def eligible_count(self, relation: int, address: int, before: int) -> int:
        if min(relation, address, before) < 0:
            raise ValueError('invalid query')
        xs = self.postings.get((relation, address), ())
        # Fast path for streaming (current time after current writer insertion).
        if before > self.last_position:
            return len(xs)
        if before == self.last_position:
            return len(xs) - int(bool(xs) and xs[-1] == before)
        return bisect_left(xs, before)

    def page_count(self, relation: int, address: int, before: int) -> int:
        n = self.eligible_count(relation, address, before)
        return (n + self.cap - 1) // self.cap

    def lookup_page(self, relation: int, address: int, before: int, page: int = 0) -> PagedRead:
        if page < 0:
            raise ValueError('negative page')
        xs = self.postings.get((relation, address), ())
        n = self.eligible_count(relation, address, before)
        pages = (n + self.cap - 1) // self.cap
        if page >= pages:
            return PagedRead((), n, pages, None, n)
        end = n - page * self.cap
        start = max(0, end - self.cap)
        candidates = tuple(reversed(xs[start:end]))
        return PagedRead(candidates, n-len(candidates), pages, page, n)

    def lookup_bits(self, relation: int, address: int, before: int,
                    bits: Sequence[int]) -> PagedRead:
        return self.lookup_page(relation, address, before, bits_page(bits))

    def lookup_ordinal(self, relation: int, address: int, before: int,
                       rank: int) -> int | None:
        """Proof/debug oracle. The neural model must NOT receive rank labels."""
        page, slot = rank_address(rank, self.cap)
        selected = self.lookup_page(relation, address, before, page)
        if slot >= len(selected.candidates):
            return None
        return selected.candidates[slot]

