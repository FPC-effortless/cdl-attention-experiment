"""Episode generation for the Phase 1 structural-routing falsification test.

Important benchmark invariant: the fixed candidate substrate A is a genuine
superset of the episode program.  Node existence is independent from edge
selection, so a router cannot solve the task by merely copying an existence
mask.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from .grammar import ARITY, Edge, Node, Op, relation_for_port


@dataclass(frozen=True)
class Episode:
    nodes: Tuple[Node, ...]
    candidate_edges: Tuple[Edge, ...]
    true_edges: Tuple[Edge, ...]
    inputs: Tuple[int, ...]
    output: int
    truth_table: Tuple[int, ...]

    @property
    def existence_mask(self) -> Tuple[int, ...]:
        return tuple(1 for _ in self.nodes)

    @property
    def true_edge_set(self) -> frozenset[Tuple[int, int, int]]:
        return frozenset((e.src, e.dst, e.port) for e in self.true_edges)


def _apply(op: Op, values: List[int]) -> int:
    if op is Op.AND:
        return int(values[0] & values[1])
    if op is Op.OR:
        return int(values[0] | values[1])
    if op is Op.XOR:
        return int(values[0] ^ values[1])
    if op is Op.NOT:
        return int(1 - values[0])
    raise ValueError(f"not an executable operation: {op}")


def _candidate_edges(nodes: List[Node]) -> List[Edge]:
    edges: List[Edge] = []
    for dst_node in nodes:
        if dst_node.op is Op.INPUT:
            continue
        for port in range(dst_node.arity):
            relation = relation_for_port(port)
            for src in range(dst_node.index):
                edges.append(Edge(src, dst_node.index, relation, port))
    return edges


def _sample_program(nodes: List[Node], rng: random.Random) -> List[Edge]:
    true_edges: List[Edge] = []
    for node in nodes:
        if node.op is Op.INPUT:
            continue
        sources = rng.sample(range(node.index), node.arity)
        for port, src in enumerate(sources):
            true_edges.append(Edge(src, node.index, relation_for_port(port), port))
    return true_edges


def _evaluate(nodes: List[Node], true_edges: List[Edge], assignment: Tuple[int, ...], output: int) -> int:
    values: Dict[int, int] = {i: assignment[i] for i, n in enumerate(nodes) if n.op is Op.INPUT}
    incoming: Dict[int, List[Edge]] = {}
    for edge in true_edges:
        incoming.setdefault(edge.dst, []).append(edge)
    for node in nodes:
        if node.op is Op.INPUT:
            continue
        edges = sorted(incoming[node.index], key=lambda e: e.port)
        if len(edges) != node.arity:
            raise ValueError(f"node {node.index} has {len(edges)} inputs; expected {node.arity}")
        values[node.index] = _apply(node.op, [values[e.src] for e in edges])
    return values[output]


def _truth_table(nodes: List[Node], true_edges: List[Edge], inputs: List[int], output: int) -> Tuple[int, ...]:
    rows = []
    for mask in range(1 << len(inputs)):
        assignment = [0] * len(nodes)
        for bit, node_index in enumerate(inputs):
            assignment[node_index] = (mask >> bit) & 1
        rows.append(_evaluate(nodes, true_edges, tuple(assignment), output))
    return tuple(rows)


def generate_episode(
    *,
    seed: int = 0,
    n_inputs: int = 3,
    n_ops: int = 4,
    output_op: Op | None = None,
) -> Episode:
    """Generate one Boolean DAG with a strict candidate-edge superset.

    Inputs occupy the first slots and operation nodes follow in topological
    order.  Every earlier node is structurally admissible to every later
    operation port.  The sampled program chooses exactly the required arity.
    """

    if n_inputs < 2 or n_ops < 1:
        raise ValueError("need at least two inputs and one operation")
    rng = random.Random(seed)
    ops = [Op.AND, Op.OR, Op.XOR, Op.NOT]
    chosen_ops = [rng.choice(ops) for _ in range(n_ops)]
    if output_op is not None:
        chosen_ops[-1] = output_op

    nodes: List[Node] = [
        Node(i, Op.INPUT, role="input", input_name=f"x{i}") for i in range(n_inputs)
    ]
    for j, op in enumerate(chosen_ops, start=n_inputs):
        nodes.append(Node(j, op, role=f"op-{op.value.lower()}"))

    true_edges = _sample_program(nodes, rng)
    candidates = _candidate_edges(nodes)
    output = len(nodes) - 1
    table = _truth_table(nodes, true_edges, list(range(n_inputs)), output)

    if not set((e.src, e.dst, e.port) for e in true_edges).issubset(
        set((e.src, e.dst, e.port) for e in candidates)
    ):
        raise AssertionError("true graph must be a subset of candidate substrate")
    return Episode(tuple(nodes), tuple(candidates), tuple(true_edges), tuple(range(n_inputs)), output, table)
