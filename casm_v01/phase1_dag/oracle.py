from itertools import combinations, product
from typing import Iterable

from .generator import Episode
from .grammar import Edge, Op


def _apply(op: Op, values):
    if op is Op.AND:
        return values[0] & values[1]
    if op is Op.OR:
        return values[0] | values[1]
    if op is Op.XOR:
        return values[0] ^ values[1]
    if op is Op.NOT:
        return 1 - values[0]
    raise ValueError(op)


def evaluate_episode(episode: Episode, edges: Iterable[Edge] | None = None):
    selected = tuple(episode.true_edges if edges is None else edges)
    incoming = {}
    for edge in selected:
        incoming.setdefault(edge.dst, []).append(edge)
    for node in episode.nodes:
        if node.op is Op.INPUT:
            continue
        got = incoming.get(node.index, [])
        if len(got) != node.arity or {e.port for e in got} != set(range(node.arity)):
            raise ValueError(f"invalid selected wiring for node {node.index}")

    outputs = []
    for bits in product((0, 1), repeat=len(episode.inputs)):
        values = dict(zip(episode.inputs, bits))
        for node in episode.nodes:
            if node.op is Op.INPUT:
                continue
            got = sorted(incoming[node.index], key=lambda e: e.port)
            values[node.index] = _apply(node.op, [values[e.src] for e in got])
        outputs.append(values[episode.output])
    return tuple(outputs)


def exhaustive_minimality(episode: Episode, max_edges: int = 12) -> dict:
    """Prove minimality/uniqueness only by exhaustive domain + subgraph search."""
    candidates = tuple(episode.candidate_edges)
    target_size = len(episode.true_edges)
    if len(candidates) > max_edges:
        return {"checked": False, "minimal": None, "unique_minimal": None,
                "reason": f"candidate substrate has {len(candidates)} edges"}
    target = episode.truth_table
    true_set = episode.true_edge_set
    for size in range(target_size):
        for subset in combinations(candidates, size):
            try:
                if evaluate_episode(episode, subset) == target:
                    return {"checked": True, "minimal": False, "unique_minimal": False,
                            "counterexample": subset}
            except (KeyError, ValueError):
                pass
    for subset in combinations(candidates, target_size):
        key = frozenset((e.src, e.dst, e.port) for e in subset)
        if key != true_set:
            try:
                if evaluate_episode(episode, subset) == target:
                    return {"checked": True, "minimal": True, "unique_minimal": False,
                            "counterexample": subset}
            except (KeyError, ValueError):
                pass
    return {"checked": True, "minimal": True, "unique_minimal": True}
