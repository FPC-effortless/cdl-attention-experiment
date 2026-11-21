"""Dispatch from a masked map with authoritative known-cell versions."""
from dataclasses import replace
from tasks import Task,MOVES
from panel import ROOT,edges,distance
from run_pilot import generate
from skills import View
from shifts import bind
from search import solve
import json

def dispatch(binding,view,goal,position):
    # Check immediately before movement in this serialized simulator.
    path=binding.read(view,goal)
    if not path:return None
    dy,dx=dict(MOVES)[path[0]]
    nxt=position[0]+dy,position[1]+dx
    return nxt if nxt in view.cells and nxt not in view.hidden else None

def run():
    rows=[]
    for seed in (9101,9107,9113):
        for trial in range(6):
            m,after=generate('maze_shift',seed,trial,{})
            t=Task(f'partial/{seed}/{trial}','maze',m.start,m.target,m.open_cells,m.size)
            view=View(t.cells);p=m.shortest().path;active=bind(t,p,view)
            block,=m.open_cells-after.open_cells
            # Authority withholds the changed cell. Planner gets only confirmed-open cells.
            view.hidden.add(block);view.change(block)
            assert dispatch(active,view,t.target,t.start) is None
            # Mask a deterministic additional area. The planner never enumerates its neighbors in the real maze.
            unknown={cell for cell in m.open_cells if (cell[0]*3+cell[1]+trial)%7==0 and cell not in (t.start,t.target)}
            view.hidden.update(unknown)
            observed=replace(t,cells=frozenset(view.cells-view.hidden))
            physical=replace(t,cells=after.open_cells)
            try:reachable=distance(observed)
            except ValueError:reachable=None
            actions=[];s=t.start;used_sensor=False
            if reachable is None:
                # Explicit reobservation of every masked cell; no movements until the fresh view arrives.
                view.hidden.clear();used_sensor=True;observed=replace(observed,cells=frozenset(view.cells))
            route=solve(observed,arm='efficient')['actions'];opt=distance(physical)
            for i,a in enumerate(route):
                b=bind(replace(observed,start=s),route[i:],view)
                nxt=dispatch(b,view,t.target,s)
                assert nxt==dict(edges(physical,s))[a] and nxt not in view.hidden
                s=nxt;actions.append(a)
            assert s==t.target
            rows.append(dict(seed=seed,trial=trial,stale_dispatches=0,masked=len(unknown|{block}),
                        visible_only_distance=reachable,reobserved=used_sensor,
                        observations=len(unknown|{block}) if used_sensor else 0,steps=len(actions),omniscient_steps=opt))
    return rows
if __name__=='__main__':
    (ROOT/'results/partial.json').write_text(json.dumps(run(),separators=(',',':'))+'\n')
