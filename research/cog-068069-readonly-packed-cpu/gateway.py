"""Unit-testable exact advisory boundary; does not read a model, network, or world."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Authority:
    lane: str
    scope: str
    tick: int
    sequence: int
    body: object

def admissible(events, scope, tick, histories):
    """Lock trusted route/occupancy before reading any eligible payload."""
    positions={}; latest={}
    for e in events:
        if e.scope!=scope or e.tick>tick:
            continue
        k=(e.lane,e.tick,e.sequence)
        if k in positions and positions[k] is not e:
            raise ValueError("ambiguous same-position authority")
        positions[k]=e
        old=latest.get(e.lane)
        if old is None or (e.tick,e.sequence)>(old.tick,old.sequence):
            latest[e.lane]=e
    out={}
    for lane,e in sorted(latest.items()):
        if not isinstance(e.body,(tuple,list)) or len(e.body)!=2:
            continue
        v,flag=e.body
        if type(v) is not int or v!=1 or type(flag) is not bool or not flag or lane not in histories:
            continue
        h=histories[lane]
        if len(h)!=3 or any(h[i][0]>=h[i+1][0] for i in range(2)) or h[-1][0]!=tick:
            continue
        out[lane]=tuple(tuple(row) for _,row in h)
    return out

def select(scores, admitted):
    if set(scores)!=set(admitted) or any(type(x) not in (int,float) or not isfinite(x) for x in scores.values()):
        raise ValueError("predictor gave malformed scores")
    return min(scores,key=lambda lane:(-scores[lane],lane)) if scores else None

def serialize(lane, admitted):
    if lane is not None and lane not in admitted:
        raise ValueError("unauthorized advisory lane")
    return {'kind':'prediction_only_not_observed','lane':lane}
