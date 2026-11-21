"""Seeded inputs and independent goal conditions for three 086 puzzle domains."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

BASE=Path(__file__).resolve().parent.parent/'cog-081-086'
sys.path.insert(0,str(BASE))
from puzzles import Maze, make_maze, Sliding
from bridge.cog066_source import acquire
from bridge.experiment_086 import scramble_options

SEEDS={'train':(8711,8719,8731),'dev':(8819,8821),'held':(8923,8929,8933),'rep':(9001,9011,9013)}
DOMAINS=('maze','slide','hanoi')
MOVES=(('U',(-1,0)),('D',(1,0)),('L',(0,-1)),('R',(0,1)))

@dataclass(frozen=True)
class Task:
    id:str
    kind:str
    start:tuple
    target:tuple
    cells:frozenset=frozenset()
    size:int=0
    pegs:int=0
    links:tuple=()

    def neighbors(self,s):
        if self.kind=='maze':
            return [(a,(s[0]+dy,s[1]+dx)) for a,(dy,dx) in MOVES if (s[0]+dy,s[1]+dx) in self.cells]
        if self.kind=='slide':
            return list(Sliding.neighbors(None,s))
        top={}
        for i,p in enumerate(s):top.setdefault(p,i)
        out=[]
        for a,i in top.items():
            for b in range(self.pegs):
                if (a,b) in self.links and (b not in top or i<top[b]):
                    z=list(s);z[i]=b;out.append(((a,b),tuple(z)))
        return out

    def h(self,s):
        if self.kind=='maze':return sum(abs(a-b) for a,b in zip(s,self.target))
        if self.kind=='hanoi':return 0
        return sum(abs(i//3-j//3)+abs(i%3-j%3) for i,v in enumerate(s) if v for j in [self.target.index(v)])

    def lower(self,s):
        if self.kind!='hanoi':return self.h(s)
        diff=[i for i,(a,b) in enumerate(zip(s,self.target)) if a!=b]
        if not diff:return 0
        k=max(diff)+1
        if self.pegs==3 and len(set(s[:k]))==len(set(self.target[:k]))==1:
            return 2**k-1
        return len(diff)

    def digest(self):
        return sha256(json.dumps([self.kind,self.start,self.target,sorted(self.cells),self.links,self.pegs]).encode()).hexdigest()

def walk(task,s,rng,n):
    prev=None
    for _ in range(n):
        choices=[z for _,z in task.neighbors(s)]
        nxt=rng.choice([z for z in choices if z!=prev] or choices)
        prev,s=s,nxt
    return s

def tasks(part):
    out=[]
    for seed in SEEDS[part]:
        for kind in DOMAINS:
            for i in range(8):
                rng=random.Random(f'{seed}/{kind}/{i}')
                name=f'{part}/{seed}/{kind}/{i}'
                if kind=='maze':
                    size=rng.choice([11,13,15] if part in ('train','dev') else [13,15,17])
                    m=make_maze(rng.randrange(10**8),size,extra=rng.choice([0,.06,.16]))
                    s,g=rng.sample(sorted(m.open_cells),2)
                    t=Task(name,kind,s,g,m.open_cells,size)
                elif kind=='slide':
                    canonical=tuple(range(1,9))+(0,)
                    proto=Task(name,kind,canonical,canonical)
                    # Independent goal walks; two nonoverlapping blank-position strata.
                    while True:
                        g=walk(proto,canonical,rng,rng.randrange(8,24))
                        if (g.index(0) in (0,1,2))==(part in ('held','rep')):break
                    s=walk(proto,g,rng,rng.randrange(7,14))
                    if s==g:s=proto.neighbors(g)[0][1]
                    t=replace(proto,start=s,target=g)
                else:
                    pegs=4 if part in ('held','rep') and i%2 else 3
                    n=rng.choice([3,4,5])
                    links=tuple((a,b) for a in range(pegs) for b in range(pegs) if a!=b and
                                (i%3==0 or abs(a-b)==1 or (pegs==4 and {a,b}=={0,3})))
                    # Relabel rods; legality remains a function of the physical topology.
                    perm=list(range(pegs));rng.shuffle(perm);links=tuple(sorted((perm[a],perm[b]) for a,b in links))
                    s=tuple(rng.randrange(pegs) for _ in range(n))
                    proto=Task(name,kind,s,s,pegs=pegs,links=links)
                    g=walk(proto,s,rng,rng.randrange(6,25))
                    if g==s:g=proto.neighbors(s)[0][1]
                    t=replace(proto,target=g)
                out.append(t)
    return out

def encoding(t,s):
    if t.kind=='slide':
        ranks={v:i+1 for i,v in enumerate(t.target) if v}
        return str((t.target.index(0),tuple(ranks.get(v,0) for v in s))),{}
    seq=list(dict.fromkeys((*t.target,*s,*range(t.pegs))))
    ids={v:i for i,v in enumerate(seq)}
    return str((t.pegs,tuple(ids[v] for v in t.target),tuple(ids[v] for v in s))),ids

def replay(t,actions):
    s=t.start
    for a in actions:
        s=dict(t.neighbors(s)).get(a)
        if s is None:return False
    return s==t.target
