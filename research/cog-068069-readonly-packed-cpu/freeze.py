"""Data construction, no learner. Freeze only after separate approval."""
import hashlib,json
from pathlib import Path
import numpy as np
from forage import emit_view
from plant061 import episode
ROOT=Path(__file__).resolve().parent
TRAIN=range(68000,68096);CAL=range(69000,69032);DEV=range(70000,70096);DESIGN=range(71000,71032)
def sensor(group,lane):
 o,_,_,_=episode(group*4+lane,n=128,shift=bool(group%2))
 return o
def cutoff(g):return 49+g%16
def main():
 out=ROOT/'frozen';out.mkdir(exist_ok=False)
 vals=[sensor(g,l)[cutoff(g)+1,2] for g in DESIGN for l in range(3)]
 threshold=float(np.quantile(vals,.72))
 train=[];cal=[];queries=[];truth={}
 for gs,dest in [(TRAIN,train),(CAL,cal)]:
  for g in gs:
   dest.append({'id':str(g),'role':'fit' if gs==TRAIN else 'cal','streams':[sensor(g,l).tolist() for l in range(3)]})
 for g in DEV:
  t=cutoff(g); lanes={chr(97+l):sensor(g,l) for l in range(3)}
  hist={k:[[i,o[i].tolist()] for i in (t-2,t-1,t)] for k,o in lanes.items()}
  for branch in (0,1):
   case=f'{g}/{branch}';scope=f'local-{g}';events=[]
   for j,k in enumerate(lanes):
    events.extend([[k,scope,t-1,1,[1,True]],[k,scope,t+1,0,{'invalid':'unavailable'}],[k,'other',t,8,{'invalid':'outside_scope'}]])
    if branch and (g%8==0 or j==g%3):events.append([k,scope,t,2,[1,False]])
   queries.append({'id':case,'group':str(g),'fault':bool(branch),'tick':t,'scope':scope,'events':events,'histories':hist})
   truth[case]={'group':str(g),'fault':bool(branch),'y':{k:int(o[t+1,2]>threshold) for k,o in lanes.items()}}
 emit_view(out,'train',train,'train',('streams',))
 emit_view(out,'cal',cal,'validation',('streams',))
 emit_view(out,'requests',queries,'validation',('events','histories'))
 (out/'truth.json').write_text(json.dumps(truth,sort_keys=True))
 (out/'fit-settings.json').write_text(json.dumps({'sensor':2,'threshold':threshold,'design_only_quantile':.72,'groups':{'train':96,'cal':32,'dev':96}}))
if __name__=='__main__':main()
