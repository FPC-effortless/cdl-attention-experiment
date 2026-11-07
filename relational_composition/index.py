"""Append-only occurrence postings with strictly causal, capped lookup."""
from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class Read:
    candidates: tuple[int, ...]
    omitted: int


class CausalIndex:
    """One document/layer per instance. Values live outside the index.

    Building postings costs O(N). Query predecessor search costs O(log bucket),
    then copies at most C references; it never scans or copies an entire bucket.
    This deliberately simple CPU reference implementation is not a GPU kernel.
    """

    def __init__(self, cap: int):
        if cap < 1:
            raise ValueError("candidate cap must be positive")
        self.cap = cap
        self.postings: dict[tuple[int, int], list[int]] = {}
        self.last_position = -1

    def append(self, position: int, relation: int, address: int) -> None:
        if position <= self.last_position or min(relation, address) < 0:
            raise ValueError("writes must be nonnegative and strictly ordered")
        self.postings.setdefault((relation, address), []).append(position)
        self.last_position = position

    def lookup(self, relation: int, address: int, before: int) -> Read:
        posting = self.postings.get((relation, address), ())
        end = bisect_left(posting, before)
        start = max(0, end - self.cap)
        return Read(tuple(reversed(posting[start:end])), start)
