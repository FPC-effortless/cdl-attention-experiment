"""Strict finite grid score and oracle envelopes; no model execution."""
from collections import Counter
import math

def strict_and_relative(logits,y,f):
    if len(logits)!=287 or not all(math.isfinite(x) for x in logits):
        raise ValueError('invalid logits')
    target=logits[y]
    return (all(target>x for i,x in enumerate(logits) if i!=y), target>logits[f])

def tally(hit):
    if len(hit)!=256:
        raise ValueError('expected 32 x 8 worlds')
    return dict(worlds=sum(hit), histories=sum(hit[i] and hit[i+1] for i in range(0,256,2)),
                families=sum(all(hit[i:i+8]) for i in range(0,256,8)))

def envelope(matrix, side):
    """matrix keyed by native + predeclared legal fixed candidates, booleans only."""
    if 'native' not in matrix or len(matrix)!=12 or len(side)!=256:
        raise ValueError('wrong matrix dimensions')
    if any(len(v)!=256 for v in matrix.values()) or set(side)!={'first','second'}:
        raise ValueError('bad matrix')
    original=matrix['native']
    candidate=[v for k,v in matrix.items() if k!='native']
    orig=tally(original)
    table={}
    for name,row in matrix.items():
        table[name]={**tally(row), 'first_worlds':sum(ok and s=='first' for ok,s in zip(row,side)),
                     'second_worlds':sum(ok and s=='second' for ok,s in zip(row,side)),
                     'worlds_rescued':sum(g and not n for g,n in zip(row,original)),
                     'worlds_damaged':sum(n and not g for g,n in zip(row,original))}
    row_oracle=[any(v[i] for v in matrix.values()) for i in range(256)]
    both=[any(v[i] and v[i+1] for v in matrix.values()) for i in range(0,256,2)]
    fam=[any(all(v[i:i+8]) for v in matrix.values()) for i in range(0,256,8)]
    if not(all(a<=b for a,b in zip(original,row_oracle)) and sum(row_oracle)>=2*sum(both)>=8*sum(fam)):
        raise ValueError('oracle lattice invariant')
    return dict(native=orig,fixed=table,
        oracle_row_worlds=sum(row_oracle),
        oracle_failed_worlds_rescued=sum(g and not n for g,n in zip(row_oracle,original)),
        oracle_failed_second_worlds_rescued=sum(g and not n and s=='second' for g,n,s in zip(row_oracle,original,side)),
        oracle_failed_first_worlds_rescued=sum(g and not n and s=='first' for g,n,s in zip(row_oracle,original,side)),
        oracle_same_gate_both_queries=sum(both),
        oracle_same_gate_all_8=sum(fam),
        oracle_four_shared_pair_gates_families=sum(all(both[i:i+4]) for i in range(0,128,4)),
        oracle_grid_only_both_queries=sum(any(v[i] and v[i+1] for v in candidate) for i in range(0,256,2)),
        any_candidate_only_cannot_rescue=sum(not any(v[i] for v in candidate) and not original[i] for i in range(256)))

if __name__=='__main__':
    orig=[True]*256;orig[0]=orig[1]=False
    x=orig.copy();x[0]=True
    y=orig.copy();y[1]=True
    sides=['first','second']*128
    m={'native':orig,'x':x,'y':y,**{f'empty{i}':[False]*256 for i in range(9)}}
    e=envelope(m,sides)
    assert e['oracle_row_worlds']==256 and e['oracle_same_gate_both_queries']==127 and e['oracle_same_gate_all_8']==31
    m['empty0']=[True]*256
    e=envelope(m,sides);assert e['oracle_same_gate_both_queries']==128 and e['oracle_same_gate_all_8']==32
    print('fixture distinguishes per-world oracle from shared-history and family oracles')
