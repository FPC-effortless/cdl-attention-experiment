"""Typed Boolean program grammar used by Phase 1."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Op(str, Enum):
    INPUT = "INPUT"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"

ARITY = {Op.INPUT: 0, Op.NOT: 1, Op.AND: 2, Op.OR: 2, Op.XOR: 2}

@dataclass(frozen=True)
class Node:
    index: int
    op: Op
    depth: int
    slot: int

    @property
    def arity(self) -> int:
        return ARITY[self.op]

@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    port: int

    @property
    def relation(self) -> int:
        # Syntactic relation only: argument port. Never an op-pair lookup.
        return self.port
