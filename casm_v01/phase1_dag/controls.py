"""Benchmark controls that validate the Phase-1 edge-selection problem."""

from __future__ import annotations

from itertools import product

from .generator import Episode
from .grammar import Edge, Op
from .oracle import evaluate_episode


def _wiring_options(episode: Episode, node_index: int) -> list[tuple[Edge, ...]]:
    node = episode.nodes[node_index]
    if node.op is Op.INPUT:
        return [()]
    sources = range(node.index)
    options = []
    # Ordered source choices correspond to the primitive's argument ports.
    for chosen in product(sources, repeat=node.arity):
        if len(set(chosen)) != node.arity:
            continue
        options.append(tuple(Edge(src, node_index, port) for port, src in enumerate(chosen)))
    return options


def valid_wirings(episode: Episode) -> list[tuple[Edge, ...]]:
    """Enumerate valid complete wirings for small benchmark instances."""
    per_node = [
        _wiring_options(episode, node.index)
        for node in episode.nodes
        if node.op is not Op.INPUT
    ]
    result: list[tuple[Edge, ...]] = []
    for choices in product(*per_node):
        result.append(tuple(edge for choice in choices for edge in choice))
    return result


def nonoracle_valid_control(episode: Episode) -> dict:
    """Find a valid wiring that differs from the oracle and changes the function."""
    target = episode.truth_table
    oracle_set = episode.true_edge_set
    checked = 0
    for wiring in valid_wirings(episode):
        key = frozenset((e.src, e.dst, e.port) for e in wiring)
        if key == oracle_set:
            continue
        checked += 1
        if evaluate_episode(episode, wiring) != target:
            return {
                "checked": checked,
                "found": True,
                "exact": False,
                "wiring": wiring,
            }
    return {"checked": checked, "found": False, "exact": None, "wiring": None}
