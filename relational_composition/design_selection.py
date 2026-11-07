"""Inference-first selection with hard competence and expressivity gates.

The 12:1 weight is a user preference, not a measured empirical law. It ranks
only *qualified* candidates and compares dimensionless log latency ratios.
The selector deliberately refuses to name a winner when no model qualifies.
"""
from dataclasses import dataclass
from math import log, isfinite
from typing import Sequence


@dataclass(frozen=True)
class Candidate:
    name: str
    inference_ms: float
    training_ms: float
    exact_accuracy: float
    causal_parity: bool
    oldest_record_accessible: bool
    memory_mib: float


def select_qualified(candidates: Sequence[Candidate], *,
                     reference_inference_ms: float,
                     reference_training_ms: float,
                     min_exact_accuracy: float=.90,
                     memory_limit_mib: float=1024,
                     inference_weight: float=12):
    if (not all(isfinite(v) for v in (min_exact_accuracy, inference_weight,
                reference_inference_ms, reference_training_ms, memory_limit_mib))
            or not 0 <= min_exact_accuracy <= 1 or inference_weight < 0
            or min(reference_inference_ms,reference_training_ms,memory_limit_mib) <= 0):
        raise ValueError('invalid selection parameters')
    results=[]
    for c in candidates:
        if (min(c.inference_ms,c.training_ms) <= 0 or c.memory_mib < 0
                or not 0 <= c.exact_accuracy <= 1
                or not all(isfinite(v) for v in (c.inference_ms,c.training_ms,
                                                  c.exact_accuracy,c.memory_mib))):
            raise ValueError('candidate must have positive finite timings')
        reasons=[]
        if not c.causal_parity: reasons.append('causality/parity not established')
        if not c.oldest_record_accessible: reasons.append('historical reachability fails')
        if c.exact_accuracy < min_exact_accuracy: reasons.append('exact-answer competence fails')
        if c.memory_mib > memory_limit_mib: reasons.append('memory ceiling exceeded')
        score=(inference_weight*log(c.inference_ms/reference_inference_ms)
               +log(c.training_ms/reference_training_ms)) if not reasons else None
        results.append(dict(name=c.name,qualified=not reasons,score=score,
                            reasons=reasons))
    eligible=[entry for entry in results if entry['qualified']]
    winner=min(eligible,key=lambda r:r['score'])['name'] if eligible else None
    return dict(winner=winner,results=results)
