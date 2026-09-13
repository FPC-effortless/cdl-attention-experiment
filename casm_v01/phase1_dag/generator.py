from dataclasses import dataclass
from itertools import product
import random
from typing import Dict, List, Tuple

from .grammar import Edge, Node, Op


@dataclass(frozen=True)
class Episode:
    nodes: Tuple[Node, ...]
    candidate_edges: Tuple[Edge, ...]
    true_edges: Tuple[Edge, ...]
    inputs: Tuple[int, ...]
    output: int
    truth_table: Tuple[int, ...]

    @property
    def true_edge_set(self):
        return frozenset((e.src, e.dst, e.port) for e in self.true_edges)


def _apply(op: Op, values: List[int]) -> int:
    if op is Op.AND:
        return values[0] & values[1]
    if op is Op.OR:
        return values[0] | values[1]
    if op is Op.XOR:
        return values[0] ^ values[1]
    if op is Op.NOT:
        return 1 - values[0]
    raise ValueError(op)


def _candidate_edges(nodes: List[Node]) -> List[Edge]:
    return [
        Edge(src, dst.index, port)
        for dst in nodes
        if dst.op is not Op.INPUT
        for port in range(dst.arity)
        for src in range(dst.index)
    ]


def _sample_true_edges(nodes: List[Node], rng: random.Random) -> List[Edge]:
    edges = []
    for node in nodes:
        if node.op is Op.INPUT:
            continue
        for port, src in enumerate(rng.sample(range(node.index), node.arity)):
            edges.append(Edge(src, node.index, port))
    return edges


def _evaluate(nodes: List[Node], edges: List[Edge], assignment: Tuple[int, ...], output: int) -> int:
    incoming: Dict[int, List[Edge]] = {}
    for edge in edges:
        incoming.setdefault(edge.dst, []).append(edge)
    values = {i: assignment[i] for i, node in enumerate(nodes) if node.op is Op.INPUT}
    for node in nodes:
        if node.op is Op.INPUT:
            continue
        selected = sorted(incoming.get(node.index, []), key=lambda e: e.port)
        if len(selected) != node.arity or {e.port for e in selected} != set(range(node.arity)):
            raise ValueError(f"invalid wiring for node {node.index}")
        values[node.index] = _apply(node.op, [values[e.src] for e in selected])
    return values[output]


def generate_episode(*, seed: int = 0, n_inputs: int = 2, n_ops: int = 2) -> Episode:
    if n_inputs < 2 or n_ops < 1:
        raise ValueError("need at least two inputs and one operation")
    rng = random.Random(seed)
    ops = [Op.AND, Op.OR, Op.XOR, Op.NOT]
    nodes = [Node(i, Op.INPUT, "input", f"x{i}") for i in range(n_inputs)]
    nodes += [Node(i, rng.choice(ops), "operation") for i in range(n_inputs, n_inputs + n_ops)]
    true_edges = _sample_true_edges(nodes, rng)
    candidates = _candidate_edges(nodes)
    rows = []
    for assignment in product((0, 1), repeat=n_inputs):
        rows.append(_evaluate(nodes, true_edges, assignment, len(nodes) - 1))
    return Episode(tuple(nodes), tuple(candidates), tuple(true_edges), tuple(range(n_inputs)), len(nodes) - 1, tuple(rows))
