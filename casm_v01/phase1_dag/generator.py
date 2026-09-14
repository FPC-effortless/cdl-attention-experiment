"""Random typed Boolean programs with a fixed candidate-edge superset."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
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

    @property
    def true_edge_set(self):
        return frozenset((e.src, e.dst, e.port) for e in self.true_edges)

    @property
    def existence(self):
        return tuple(1 if i < len(self.nodes) else 0 for i in range(len(self.nodes)))

class BooleanDAGGenerator:
    """Generate programs while keeping A fixed as an upper-triangular superset.

    A is therefore not the oracle wiring: every earlier node is a candidate source.
    The per-episode existence mask only removes unused slots; it cannot identify edges.
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
        edges: list[Edge] = []
        for i in range(input_count, n):
            op = self.rng.choice(OPS)
            parents = self.rng.sample(range(i), k=1 if op is Op.NOT else 2)
            child_depth = 1 + max(nodes[p].depth for p in parents)
            nodes.append(Node(i, op, child_depth, i))
            edges.extend(Edge(p, i, port) for port, p in enumerate(parents))
        # Fixed candidate substrate: all earlier nodes may feed every later node,
        # subject only to arity. This creates genuine distractor edges.
        candidates = tuple(Edge(src, dst, port)
                           for dst in range(input_count, n)
                           for port in range(nodes[dst].arity)
                           for src in range(dst))
        vals = tuple(self.rng.randint(0, 1) for _ in range(input_count))
        target = self._eval(nodes, edges, vals, n - 1)
        table = {}
        for bits in __import__('itertools').product((0, 1), repeat=input_count):
            table[bits] = self._eval(nodes, edges, bits, n - 1)
        return Episode(nodes, tuple(edges), candidates, tuple(range(input_count)), n - 1, vals, target, table)

    @staticmethod
    def _eval(nodes, edges, inputs, output):
        values = {i: int(v) for i, v in enumerate(inputs)}
        incoming: dict[int, dict[int, int]] = {}
        for e in edges:
            incoming.setdefault(e.dst, {})[e.port] = e.src
        for node in nodes:
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
