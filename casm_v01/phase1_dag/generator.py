from dataclasses import dataclass
from itertools import combinations, product
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


def _truth_table(nodes: List[Node], edges: List[Edge], n_inputs: int, output: int) -> Tuple[int, ...]:
    return tuple(
        _evaluate(nodes, edges, assignment, output)
        for assignment in product((0, 1), repeat=n_inputs)
    )


def _valid_wirings(nodes: List[Node]) -> List[Tuple[Edge, ...]]:
    per_node = []
    for node in nodes:
        if node.op is Op.INPUT:
            continue
        options = []
        for chosen in product(range(node.index), repeat=node.arity):
            if len(set(chosen)) != node.arity:
                continue
            options.append(tuple(
                Edge(src, node.index, port) for port, src in enumerate(chosen)
            ))
        per_node.append(options)
    return [
        tuple(edge for choice in choices for edge in choice)
        for choices in product(*per_node)
    ]


def _is_selection_problem(nodes: List[Node], true_edges: List[Edge],
                          truth_table: Tuple[int, ...], n_inputs: int, output: int) -> bool:
    """Require a functionally discriminative and globally unique minimal oracle."""
    candidates = _candidate_edges(nodes)
    true_key = frozenset((e.src, e.dst, e.port) for e in true_edges)
    wirings = _valid_wirings(nodes)

    # At least one alternative valid wiring must change the Boolean function.
    if not any(
        frozenset((e.src, e.dst, e.port) for e in wiring) != true_key
        and _truth_table(nodes, list(wiring), n_inputs, output) != truth_table
        for wiring in wirings
    ):
        return False

    # Prove global minimality and uniqueness for the small exhaustive regime.
    target_size = len(true_edges)
    for size in range(target_size):
        for subset in combinations(candidates, size):
            try:
                if _truth_table(nodes, list(subset), n_inputs, output) == truth_table:
                    return False
            except (KeyError, ValueError):
                pass
    for subset in combinations(candidates, target_size):
        key = frozenset((e.src, e.dst, e.port) for e in subset)
        if key == true_key:
            continue
        try:
            if _truth_table(nodes, list(subset), n_inputs, output) == truth_table:
                return False
        except (KeyError, ValueError):
            pass
    return True


def generate_episode(*, seed: int = 0, n_inputs: int = 2, n_ops: int = 2) -> Episode:
    if n_inputs < 2 or n_ops < 1:
        raise ValueError("need at least two inputs and one operation")
    if n_inputs > 4 or n_ops > 3:
        raise ValueError("Phase-1 exhaustive validity generation is limited to small graphs")

    rng = random.Random(seed)
    ops = [Op.AND, Op.OR, Op.XOR, Op.NOT]
    output = n_inputs + n_ops - 1

    # Rejection sampling is part of the benchmark definition: every emitted
    # episode is itself a valid edge-selection problem, not a post-hoc filter.
    for _attempt in range(512):
        nodes = [Node(i, Op.INPUT, "input", f"x{i}") for i in range(n_inputs)]
        nodes += [Node(i, rng.choice(ops), "operation")
                  for i in range(n_inputs, n_inputs + n_ops)]
        true_edges = _sample_true_edges(nodes, rng)
        truth = _truth_table(nodes, true_edges, n_inputs, output)
        if _is_selection_problem(nodes, true_edges, truth, n_inputs, output):
            return Episode(
                tuple(nodes), tuple(_candidate_edges(nodes)), tuple(true_edges),
                tuple(range(n_inputs)), output, truth,
            )

    raise RuntimeError(
        f"could not sample a non-degenerate Phase-1 selection problem for seed {seed}"
    )
