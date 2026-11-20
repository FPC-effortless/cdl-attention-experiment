import time
from observe import write
"""Emit complete held predictions and time; reads no truth or simulator code."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from model import ARMS,encode,read_view,prepare,rollout
if sys.flags.optimize:raise RuntimeError('unsafe optimized execution')
OUT=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
def model(a):
 a='predictive' if a=='persistence' else a.split('-')[0]
 with np.load(a+'.npz',allow_pickle=False) as m:return dict(m)
def main():
 started=time.perf_counter()
 assert not Path('truth.json').exists() and not Path('train.json').exists()
 data=read_view(Path('challenge.json')); assert len(data)==64
 plans=np.zeros((4,16,2)); plans[:,:4,:]=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])[:,None,:]*.65
 ms={arm:model(arm) for arm in ARMS}; preds={};times=[]
 probes={arm:{} for arm in ('predictive-adaptive','pca-adaptive','narx-adaptive')}
 for rep in range(3):
  for a in ARMS[rep:]+ARMS[:rep]:
   m=ms[a];arm={};sec=0.
   for id,(o,u) in data.items():
    arm[id]={}
    for s in (0,4,16):
     t=time.perf_counter();h,beta=prepare(m,o[:41+s],u[:40+s],a); y=rollout(m,h,beta,plans,a); sec+=time.perf_counter()-t
     if rep==0:arm[id][s]=y.tolist()
    if rep==0 and a in probes:probes[a][id]=encode(m,o).tolist()
   if rep==0:preds[a]=arm
   times.append({'arm':a,'repeat':rep,'whole_workload_seconds':sec,'requests':192,'four_branches_per_request':True})
 for key,val in [('predictions',preds),('timing',times),('held-encodings',probes),('metrics',{'requests_per_arm':192})]:
  (OUT/(key+'.json')).write_text(json.dumps(val,allow_nan=False))
 write({"complete":len(times),"groups":len(data),"model_size":max(sum(np.size(x) for k,x in m.items() if k!="gram_cond") for m in ms.values()),"work_seconds":sum(x["whole_workload_seconds"] for x in times)})
if __name__=='__main__':main()
