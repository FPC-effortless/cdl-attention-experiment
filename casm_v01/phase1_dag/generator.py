"""Random typed Boolean programs with a fixed candidate-edge superset."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import random
from .grammar import Edge, Node, Op

OPS = (Op.NOT, Op.AND, Op.OR, Op.XOR)

@dataclass
class Episode:
    nodes: list[Node]
    true_edges: tuple[Edge, ...]
    candidate_edges: tuple[Edge, ...]
    inputs: tuple[int, ...]
    output: int
    input_values: tuple[int, ...]
    target: int
    truth_table: dict[tuple[int, ...], int]
    active_count: int

    @property
    def true_edge_set(self):
        return frozenset((e.src, e.dst, e.port) for e in self.true_edges)

    @property
    def existence_mask(self):
        return [1.0 if i < self.active_count else 0.0 for i in range(len(self.nodes))]

class BooleanDAGGenerator:
    """Generate programs on one fixed slot substrate.

    A is the same upper-triangular candidate substrate for every episode. The
    existence mask removes unused slots, while true wiring is one of many
    admissible wirings; m_i*m_j*A_ij is therefore not the oracle.
    """
    def __init__(self, max_nodes: int = 10, min_nodes: int = 4, seed: int = 0):
        if not (2 <= min_nodes <= max_nodes):
            raise ValueError("invalid node range")
        self.max_nodes, self.min_nodes = max_nodes, min_nodes
        self.rng = random.Random(seed)

    def _make(self, n: int, input_count: int | None = None) -> Episode:
        if input_count is None:
            input_count = self.rng.randint(2, min(4, n - 1))
        nodes = [Node(i, Op.INPUT, 0, i) for i in range(input_count)]
        true_edges: list[Edge] = []
        for i in range(input_count, n):
            op = self.rng.choice(OPS)
            parents = self.rng.sample(range(i), k=1 if op is Op.NOT else 2)
            depth = 1 + max(nodes[p].depth for p in parents)
            nodes.append(Node(i, op, depth, i))
            true_edges.extend(Edge(p, i, port) for port, p in enumerate(parents))
        nodes.extend(Node(i, Op.INPUT, 0, i) for i in range(n, self.max_nodes))
        candidates = tuple(
            Edge(src, dst, port)
            for dst in range(input_count, n)
            for port in range(nodes[dst].arity)
            for src in range(dst)
        )
        vals = tuple(self.rng.randint(0, 1) for _ in range(input_count))
        target = self._eval(nodes, true_edges, vals, n - 1)
        table = {bits: self._eval(nodes, true_edges, bits, n - 1)
                 for bits in product((0, 1), repeat=input_count)}
        return Episode(nodes, tuple(true_edges), candidates, tuple(range(input_count)),
                       n - 1, vals, target, table, n)

    @staticmethod
    def _eval(nodes, edges, inputs, output):
        values = {i: int(v) for i, v in enumerate(inputs)}
        incoming: dict[int, dict[int, int]] = {}
        for e in edges:
            incoming.setdefault(e.dst, {})[e.port] = e.src
        for node in nodes:
            if node.index > output:
                break
            if node.op is Op.INPUT:
                continue
            a = [values[incoming[node.index][p]] for p in range(node.arity)]
            if node.op is Op.NOT: values[node.index] = 1 - a[0]
            elif node.op is Op.AND: values[node.index] = a[0] & a[1]
            elif node.op is Op.OR: values[node.index] = a[0] | a[1]
            elif node.op is Op.XOR: values[node.index] = a[0] ^ a[1]
        return values[output]

    def sample(self) -> Episode:
        return self._make(self.rng.randint(self.min_nodes, self.max_nodes))

    def sample_batch(self, count: int) -> list[Episode]:
        return [self.sample() for _ in range(count)]
