"""Oracle execution and exhaustive small-domain graph checks."""

from itertools import combinations, product
from typing import Iterable, List, Sequence, Tuple

from .generator import Episode
from .grammar import Op, Edge


def _apply(op: Op, values: Sequence[int]) -> int:
    if op is Op.AND:
        return int(values[0] & values[1])
    if op is Op.OR:
        return int(values[0] | values[1])
    if op is Op.XOR:
        return int(values[0] ^ values[1])
    if op is Op.NOT:
        return int(1 - values[0])
    raise ValueError(op)


def evaluate_episode(episode: Episode, edges: Iterable[Edge] | None = None) -> Tuple[int, ...]:
    """Evaluate a complete candidate graph on every Boolean input assignment.

    This oracle is deliberately strict: each operation must receive exactly
    its declared arity.  It therefore exposes the copy-mask control rather
    than silently changing the primitive semantics when distractor edges are
    present.
    """

    selected = tuple(episode.true_edges if edges is None else edges)
    incoming = {}
    for edge in selected:
        incoming.setdefault(edge.dst, []).append(edge)

    for node in episode.nodes:
        if node.op is Op.INPUT:
            continue
        got = incoming.get(node.index, [])
        if len(got) != node.arity:
            raise ValueError(
                f"node {node.index} ({node.op.value}) has {len(got)} selected inputs; "
                f"expected {node.arity}"
            )
        if len({e.port for e in got}) != node.arity:
            raise ValueError(f"node {node.index} has duplicate/missing input ports")

    outputs: List[int] = []
    for bits in product((0, 1), repeat=len(episode.inputs)):
        values = {node: bit for node, bit in zip(episode.inputs, bits)}
        for node in episode.nodes:
            if node.op is Op.INPUT:
                continue
            got = sorted(incoming[node.index], key=lambda e: e.port)
            values[node.index] = _apply(node.op, [values[e.src] for e in got])
        outputs.append(values[episode.output])
    return tuple(outputs)


def equivalent_to_target(episode: Episode, edges: Iterable[Edge]) -> bool:
    try:
        return evaluate_episode(episode, edges) == episode.truth_table
    except (KeyError, ValueError):
        return False


def exhaustive_minimality(episode: Episode, max_edges: int = 12) -> dict:
    """Check whether any smaller admissible graph computes the same function.

    Exhaustive search is intentionally restricted to small episodes.  A
    positive result proves global minimality within the enumerated candidate
    substrate; it is not replaced by a local edge-removal heuristic.
    """

    candidates = tuple(episode.candidate_edges)
    target_size = len(episode.true_edges)
    if len(candidates) > max_edges:
        return {
            "checked": False,
            "reason": f"candidate substrate has {len(candidates)} edges > max_edges={max_edges}",
            "minimal": None,
            "counterexample": None,
        }

    target = episode.truth_table
    for size in range(target_size):
        for subset in combinations(candidates, size):
            if equivalent_to_target(episode, subset):
                return {
                    "checked": True,
                    "minimal": False,
                    "counterexample": tuple(subset),
                    "searched_sizes": size + 1,
                }
    return {
        "checked": True,
        "minimal": True,
        "counterexample": None,
        "searched_sizes": target_size,
    }
