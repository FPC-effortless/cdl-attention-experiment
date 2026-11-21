"""Local provenance certificates and state-conditioned skill descriptors."""
from dataclasses import dataclass
from tasks import MOVES,encoding

@dataclass(frozen=True)
class Corridor:
    start:tuple
    end:tuple
    path:tuple
    visited:tuple
    versions:tuple

class View:
    def __init__(self,cells):
        self.cells=set(cells)
        self.revisions={}
        self.hidden=set()
        self.goal_version=0
        self.records={}
        self.counts={'reads':0,'checks':0,'hits':0,'miss':0}

    def change(self,cell):
        if cell in self.cells:self.cells.remove(cell)
        else:self.cells.add(cell)
        self.revisions[cell]=self.revisions.get(cell,0)+1

    def valid(self,record):
        for cell,rev in record.versions:
            self.counts['checks']+=1
            if cell in self.hidden or self.revisions.get(cell,0)!=rev:return False
        return True

    def corridor(self,start,action):
        key=start,action
        if key in self.records and self.valid(self.records[key]):
            self.counts['hits']+=1
            return self.records[key]
        self.counts['miss']+=1;deps={};path=[];visited=[]
        dy,dx=dict(MOVES)[action];cur=(start[0]+dy,start[1]+dx);prev=start
        for cell in (start,cur):
            deps[cell]=self.revisions.get(cell,0)
            self.counts['reads']+=1
            if cell in self.hidden or cell not in self.cells:return None
        a=action;seen={start}
        while cur not in seen:
            seen.add(cur);path.append(a);visited.append(cur)
            exits=[]
            for b,(dy,dx) in MOVES:
                cell=cur[0]+dy,cur[1]+dx
                if cell in self.hidden:return None
                deps[cell]=self.revisions.get(cell,0);self.counts['reads']+=1
                if cell in self.cells and cell!=prev:exits.append((b,cell))
            if len(exits)!=1:break
            a,nxt=exits[0];prev,cur=cur,nxt
        record=Corridor(start,visited[-1],tuple(path),tuple(visited),tuple(deps.items()))
        self.records[key]=record
        return record

def suffix(t,state,book):
    k,ids=encoding(t,state);seq=book.get(t.kind,{}).get(k)
    if not seq:return ()
    if t.kind=='slide':return tuple(seq)
    inv={v:k for k,v in ids.items()}
    return tuple((inv[a],inv[b]) for a,b in seq)

def features(t,s,neighbors,seq):
    # Features only use the current observation and training dictionary.
    return [float(t.kind==k) for k in ('maze','slide','hanoi')]+[
        len(neighbors),t.h(s),len(seq),t.size,t.pegs,len(s),
        sum(a!=b for a,b in zip(s,t.target)),int(s==t.start)]

def predict(tree,x):
    while 'leaf' not in tree:
        tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
    return tree['leaf']
