from dataclasses import dataclass
from enum import Enum


class Op(str, Enum):
    INPUT = "INPUT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"


ARITY = {Op.AND: 2, Op.OR: 2, Op.XOR: 2, Op.NOT: 1}


@dataclass(frozen=True)
class Node:
    index: int
    op: Op
    role: str
    input_name: str | None = None

    @property
    def arity(self) -> int:
        return 0 if self.op is Op.INPUT else ARITY[self.op]


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    port: int

    @property
    def relation(self) -> str:
        return f"arg{self.port + 1}-of"
