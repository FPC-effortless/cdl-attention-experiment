"""Changes are delivered through the local observation and revision interface."""
import json
from pathlib import Path
from dataclasses import dataclass,replace
from tasks import Task,MOVES,replay
from panel import distance
from search import solve
from skills import View
from run_pilot import generate

@dataclass
class Binding:
    goal:tuple
    epoch:int
    path:tuple
    deps:tuple

    def read(self,view,goal):
        if self.goal!=goal or self.epoch!=view.goal_version:return None
        for cell,v in self.deps:
            if cell in view.hidden or view.revisions.get(cell,0)!=v:return None
        return self.path

def bind(t,path,view):
    cells=[t.start];s=t.start
    for a in path:
        s=dict(t.neighbors(s))[a];cells.append(s)
    return Binding(t.target,view.goal_version,tuple(path),tuple((c,view.revisions.get(c,0)) for c in cells))

def run():
    rows=[]
    for seed in (8101,8107,8113,9101,9107,9113):
        for trial in range(6):
            m,changed=generate('maze_shift',seed,trial,{})
            t=Task(f'shift/{seed}/{trial}','maze',m.start,m.target,m.open_cells,m.size)
            p=tuple(m.shortest().path);v=View(m.open_cells);binding=bind(t,p,v)
            assert binding.read(v,t.target)==p
            # Learn local records before the edit. Include wall cells in their read sets.
            for cell in sorted(m.open_cells):
                for a,_ in t.neighbors(cell):v.corridor(cell,a)
            old=list(v.records.values());block,=m.open_cells-changed.open_cells
            v.change(block);assert binding.read(v,t.target) is None
            good=bad=0
            for r in old:
                affected=block in dict(r.versions)
                assert v.valid(r)!=affected
                if affected:bad+=1
                else:
                    good+=1
                    assert v.corridor(r.start,r.path[0]) is r
            live=replace(t,cells=frozenset(v.cells))
            assert not replay(live,p) and good and bad
            fresh=solve(live,arm='efficient')['actions'];assert len(fresh)==distance(live)
            gb=bind(live,fresh,v)
            newgoal=dict(live.neighbors(live.target)).popitem()[1]
            v.goal_version+=1;assert gb.read(v,newgoal) is None
            unaffected=next(r for r in old if v.valid(r));assert v.valid(unaffected)
            targettask=replace(live,target=newgoal);newpath=solve(targettask,arm='efficient')['actions']
            assert len(newpath)==distance(targettask)
            # Goal inside the skill requires truncation even with unchanged topology.
            long=next(r for r in v.records.values() if len(r.path)>1 and v.valid(r))
            inner=replace(live,start=long.start,target=long.visited[0])
            assert solve(inner,'structural',model={'book':{},'corridor_support':1})['steps']==1
            # Opening a neighboring wall can create a junction without invalidating the path itself.
            side=next(c for c,_ in long.versions if c not in v.cells)
            v.change(side);assert not v.valid(long)
            v.change(side)
            record=v.corridor(long.start,long.path[0]);assert record is not None
            # Lose visibility of a dependency; dispatch must not consume cached or fallback omniscient edges.
            witness=record.visited[0];v.hidden.add(witness)
            assert not v.valid(record) and v.corridor(record.start,record.path[0]) is None
            visible_task=replace(live,start=record.start,target=witness,cells=frozenset(v.cells-v.hidden))
            assert witness not in {n for _,n in visible_task.neighbors(record.start)}
            # When sensing is unavailable, defer the hidden target. When it is restored,
            # build the live view from newly observed facts and dispatch one validated edge.
            v.hidden.remove(witness)
            again=v.corridor(record.start,record.path[0])
            assert again and v.valid(again) and again.visited[0]==witness
            rows.append(dict(seed=seed,trial=trial,block=block,old_steps=len(p),new_steps=len(fresh),
                             old_rejected=True,retained=good,invalidated=bad,goal_rejected=True,
                             fresh_goal_steps=len(newpath),offpath_rejected=True,hidden_deferred=True,
                             restored=True,stats=v.counts))
    return rows
if __name__=='__main__':
    (Path(__file__).resolve().parent/'results/shifts.json').write_text(json.dumps(run(),separators=(',',':'))+'\n')
