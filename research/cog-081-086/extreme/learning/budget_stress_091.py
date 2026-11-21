"""Post hoc budget sensitivity on fresh tier 8-10 mazes (not tuning)."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from difficulty import make_maze_challenge,maze_changed
from maze_transfer_091 import explore,collect_training,calibrate,preregistered_budget

CASES=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
traces,model,options,ns=collect_training(CASES)
alpha,_=calibrate(CASES,model)
rows=[]
for seed in (9803,9809,9811,9817):
 for level in (8,9,10):
  maze,ref=make_maze_challenge(seed*100003+level*131,level)
  shifted,_=maze_changed(maze,ref.path)
  for world,instance in (('original',maze),('shift',shifted)):
   for factor in (1,2,4):
    for arm in ('heuristic','learned'):
     r=explore(instance,mode=arm,model=model,alpha=alpha,budget=preregistered_budget(instance.size)*factor)
     rows.append(dict(seed=seed,level=level,world=world,budget_factor=factor,arm=arm,
                      solved=r['solved'],actions=r['actions'],elapsed_ns=r['elapsed_ns']))
report={'label':'post hoc exploratory budget sensitivity only','trained_on':'development levels 1-5',
        'frozen_alpha':alpha,'summary':{},'rows':rows}
for factor in (1,2,4):
 report['summary'][str(factor)]={arm:{'solved':sum(r['solved'] for r in rows if r['budget_factor']==factor and r['arm']==arm),
                                     'n':sum(1 for r in rows if r['budget_factor']==factor and r['arm']==arm),
                                     'actions':sum(r['actions'] for r in rows if r['budget_factor']==factor and r['arm']==arm),
                                     'total_ms':round(sum(r['elapsed_ns'] for r in rows if r['budget_factor']==factor and r['arm']==arm)/1e6,3)}
                                  for arm in ('heuristic','learned')}
out=HERE/'results'/'maze-091-budget.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report['summary'],indent=2))
