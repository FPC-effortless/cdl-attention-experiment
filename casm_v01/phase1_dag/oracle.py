"""Exact Boolean execution and exhaustive structural checks."""
from __future__ import annotations
from itertools import product
from .grammar import Edge, Op

def evaluate_episode(episode, wiring=None, inputs=None):
    wiring = tuple(episode.true_edges if wiring is None else wiring)
    inputs = episode.input_values if inputs is None else tuple(inputs)
    values = {i: float(v) for i, v in zip(episode.inputs, inputs)}
    incoming = {}
    for e in wiring:
        incoming.setdefault(e.dst, {})[e.port] = e.src
    for node in episode.nodes[:episode.active_count]:
        if node.op is Op.INPUT:
            continue
        if any(p not in incoming.get(node.index, {}) for p in range(node.arity)):
            raise ValueError(f"incomplete wiring for node {node.index}")
        a = [values[incoming[node.index][p]] for p in range(node.arity)]
        if node.op is Op.NOT: values[node.index] = 1.0 - a[0]
        elif node.op is Op.AND: values[node.index] = a[0] * a[1]
        elif node.op is Op.OR: values[node.index] = a[0] + a[1] - a[0] * a[1]
        elif node.op is Op.XOR: values[node.index] = a[0] + a[1] - 2.0 * a[0] * a[1]
    return int(values[episode.output] >= 0.5)

def exhaustive_truth_table(episode, wiring=None):
    return {bits: evaluate_episode(episode, wiring=wiring, inputs=bits)
            for bits in product((0, 1), repeat=len(episode.inputs))}

def locally_nonredundant(episode):
    oracle = episode.true_edge_set
    for removed in episode.true_edges:
        remaining = tuple(e for e in episode.true_edges if e != removed)
        for bits, expected in episode.truth_table.items():
            try:
                got = evaluate_episode(episode, remaining, bits)
            except ValueError:
                got = None
            if got != expected:
                break
        else:
            return False
    return True
