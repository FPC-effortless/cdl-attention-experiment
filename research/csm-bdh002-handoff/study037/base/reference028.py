"""Typed transaction semantics. Parser and exact closure events are supplied."""
from dataclasses import dataclass,replace
import json,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
@dataclass(frozen=True)
class Op:
    tag:str
    d:int
    arg:int=0
@dataclass(frozen=True)
class State:
    cells:tuple
    pending:Op|None=None
    required:int|None=None

def read(cells,i):
    seen=set()
    while cells[i] is not None:
        if i in seen:raise ValueError('cycle')
        seen.add(i);tag,n=cells[i]
        if tag=='v':return n
        i=n
    return None

def wait(s,op,required=None):
    return replace(s,pending=op,required=required)
def cancel(s):
    return State(s.cells)
def commit(s):
    p=s.pending
    if p is None or (s.required is not None and read(s.cells,s.required) is None):return cancel(s)
    c=list(s.cells)
    if p.tag=='set':c[p.d]=('v',p.arg)
    elif p.tag=='delete':c[p.d]=None
    elif p.tag=='copy':
        v=read(s.cells,p.arg);c[p.d]=None if v is None else ('v',v)
    elif p.tag=='link':
        c[p.d]=('l',p.arg)
        try:read(c,p.d)
        except ValueError:return cancel(s)
    else:raise ValueError(p.tag)
    return State(tuple(c))

def baseline(registers,p,cond):
    def resolve(i):
        for _ in range(len(registers)+1):
            v=registers[i]
            if v>=-1:return v
            i=-v-2
        return -999
    if cond is not None and resolve(cond)==-1:return
    before=registers[p.d]
    if p.tag=='set':registers[p.d]=p.arg
    elif p.tag=='delete':registers[p.d]=-1
    elif p.tag=='copy':registers[p.d]=resolve(p.arg)
    else:
        registers[p.d]=-p.arg-2
        if resolve(p.d)==-999:registers[p.d]=before

def compose(first,after):
    a,u=first;b,v=after
    return b@a,b@u+v

def scan(seq):
    out=list(seq);stride=1
    while stride<len(out):
        out=out[:stride]+[compose(x,y) for x,y in zip(out,out[stride:])]
        stride*=2
    return out

def main():
    stats={}
    for horizon in (12,64,256,1024):
        count=0;r=random.Random(horizon)
        for _ in range(32):
            s=State((None,)*20);reg=[-1]*20
            for j in range(horizon):
                op=Op(r.choice(['set','copy','link','delete']),r.randrange(20),r.randrange(20))
                cond=r.choice([None,None,r.randrange(20)])
                p=wait(s,op,cond);assert p.cells is s.cells
                if r.randrange(7)==0:s=cancel(p)
                else:s=commit(p);baseline(reg,op,cond)
                assert [(-1 if v is None else v[1] if v[0]=='v' else -2-v[1]) for v in s.cells]==reg
                count+=1
        stats[str(horizon)]=count
    for depth in range(1,17):
        s=commit(wait(State((None,)*20),Op('set',0,3)))
        for i in range(1,depth+1):s=commit(wait(s,Op('link',i,i-1)))
        assert read(s.cells,depth)==3
        s=commit(wait(s,Op('copy',19,depth)))
        s=commit(wait(s,Op('set',0,5)))
        assert (read(s.cells,depth),read(s.cells,19))==(5,3)
        assert commit(wait(s,Op('link',0,depth)))==s
    # Same pending COPY; suffix authorizes it, blocks it, or cancels it.
    trap_checks=0
    for a in range(16):
        s=commit(wait(State((None,)*4),Op('set',0,a)))
        pending=wait(s,Op('copy',1,0))
        assert read(pending.cells,1) is None
        assert read(commit(pending).cells,1)==a
        assert commit(replace(pending,required=2))==s and cancel(pending)==s
        assert read(commit(replace(pending,required=0)).cells,1)==a
        trap_checks+=1
    r=np.random.default_rng(28);error=0
    for n in (1,2,7,16,65,257):
        seq=[(np.eye(3)*.8+r.normal(0,.04,(3,3)),r.normal(size=3)) for _ in range(n)]
        serial=np.array([1.,2.,3.]);initial=serial.copy()
        for (a,b),(m,u) in zip(seq,scan(seq)):
            serial=a@serial+b;error=max(error,float(np.max(abs(serial-(m@initial+u)))))
    assert error<1e-10
    (ROOT/'reference_checks.json').write_text(json.dumps(dict(differential_transitions=stats,depths=list(range(1,17)),trap_fixtures=trap_checks,max_affine_prefix_error=error),indent=2))
if __name__=='__main__':main()
