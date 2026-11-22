"""Fixed 11 simplex choices. No trained model state or text input."""
from itertools import combinations
import math
FLOOR=1e-4
N=4

def choices():
    # Distinguish old→new causal lag convention from channel order.
    groups=[(f'corner_{i}',(i,)) for i in range(N)]
    groups += [(f'mid_{i}_{j}',(i,j)) for i,j in combinations(range(N),2)]
    groups += [('uniform',tuple(range(N)))]
    out={}
    for name,loc in groups:
        v=tuple(FLOOR+(1-N*FLOOR)/len(loc) if k in loc else FLOOR for k in range(N))
        if min(v)<FLOOR or abs(math.fsum(v)-1)>1e-15:raise ValueError('bad static simplex point')
        out[name]=v
    return out
