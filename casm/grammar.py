"""Typed Boolean DAG primitives used by CASM Phase 1.

The executable program is acyclic and has a hard fan-in cap of two.  The
candidate structural substrate may contain more admissible edges; an episode's
true program selects only the edges actually used by the computation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Op(str, Enum):
    INPUT = "INPUT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"


ARITY = {Op.AND: 2, Op.OR: 2, Op.XOR: 2, Op.NOT: 1}


@dataclass(frozen=True)
class Node:
    """A node in topological order; inputs have no predecessors."""

    index: int
    op: Op
    role: str
    input_name: str | None = None

    @property
    def arity(self) -> int:
        return 0 if self.op is Op.INPUT else ARITY[self.op]


@dataclass(frozen=True)
class Edge:
    """A directed candidate edge from an earlier node to a later node."""

    src: int
    dst: int
    relation: str
    port: int


RELATIONS: Tuple[str, ...] = ("arg1-of", "arg2-of", "output-of")


def relation_for_port(port: int) -> str:
    if port == 0:
        return "arg1-of"
    if port == 1:
        return "arg2-of"
    raise ValueError(f"unsupported port: {port}")
