"""Paired structural counterfactuals with identical node-local statistics."""
from __future__ import annotations
from itertools import product
import random
from .generator import Episode, BooleanDAGGenerator
from .grammar import Edge, Op

def rewire_episode(ep: Episode, seed: int = 0) -> Episode:
    rng=random.Random(seed); edges=list(ep.true_edges); changed=False
    for node in ep.nodes[:ep.active_count]:
        if node.op is Op.INPUT: continue
        old=[e.src for e in edges if e.dst==node.index]
        choices=[i for i in range(node.index) if i not in old]
        if choices:
            for p,src in enumerate(old):
                if choices:
                    new=rng.choice(choices); edges.remove(Edge(src,node.index,p)); edges.append(Edge(new,node.index,p)); choices.remove(new); changed=True; break
        if changed: break
    if not changed: raise ValueError("could not produce structural pair")
    edges=tuple(sorted(edges,key=lambda e:(e.dst,e.port)))
    parents=list(ep.parent_slots)
    for node in ep.nodes[:ep.active_count]:
        if node.op is Op.INPUT: continue
        parents[node.index]=tuple(e.src for e in sorted((e for e in edges if e.dst==node.index),key=lambda e:e.port))
    vals=ep.input_values; target=BooleanDAGGenerator._eval(ep.nodes,edges,vals,ep.output)
    table={bits:BooleanDAGGenerator._eval(ep.nodes,edges,bits,ep.output) for bits in product((0,1),repeat=len(ep.inputs))}
    return Episode(ep.nodes,edges,ep.candidate_edges,ep.inputs,ep.output,vals,target,table,ep.active_count,tuple(parents))
