"""Canonicalization is for deduplication only; raw operand order is preserved for evaluation."""
from __future__ import annotations
from .grammar import Op

COMMUTATIVE={Op.AND,Op.OR,Op.XOR}

def canonical_edge_key(episode):
    """Return a canonical structural key without mutating the raw episode.

    Commutative ports are sorted only for uniqueness bookkeeping. Evaluation and
    permutation tests continue to use the original arg1/arg2 labels.
    """
    rows=[]
    for node in episode.nodes[:episode.active_count]:
        incoming=sorted((e.src,e.port) for e in episode.true_edges if e.dst==node.index)
        if node.op in COMMUTATIVE:
            incoming=[(src,p) for p,(src,_) in enumerate(sorted(incoming))]
        rows.append((node.index,node.op.value,node.depth,tuple(incoming)))
    return tuple(rows)
