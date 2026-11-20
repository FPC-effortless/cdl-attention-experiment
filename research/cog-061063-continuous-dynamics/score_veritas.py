import time
from observe import write
"""Experiment-specific verifier adapter. Scores saved predictions; no dynamics rerun."""
import json,os,sys
from pathlib import Path
import numpy as np
from model import ARMS,read_view,encode,ridge,LAG
if sys.flags.optimize:raise RuntimeError('unsafe optimized execution')
def read(p):return json.loads(Path(p).read_text())
def main():
 started=time.perf_counter()
 out=Path(os.environ['TRAINLAB_OUTPUT_DIR']); truth=read('truth.json'); p=read('predictions.json'); held=read('held-encodings.json')
 groups=truth['evaluation']; assert set(p)==set(ARMS) and len(groups)==64
 assert sum(v['shifted'] for v in groups.values())==32
 cells={}; sample={}
 response=[-1,0,0,1];interaction=[1,-1,-1,1]
 for a in ARMS:
  assert set(p[a])==set(groups)
  for id,g in groups.items():
   assert set(g['checks'])==set(p[a][id])=={'0','4','16'}
   for n,c in g['checks'].items():
    acts=np.array([b['u'] for b in c['branches']]);assert acts.shape==(4,16,2) and np.all(acts[:,4:]==0) and np.array_equal(acts[:,:4],np.broadcast_to(np.array([[-1,-1],[1,-1],[-1,1],[1,1]])[:,None,:]*.65,(4,4,2)))
    y=np.array([b['y'] for b in c['branches']]);pr=np.array(p[a][id][n])
    assert pr.shape==y.shape==(4,16,16) and np.isfinite(pr).all() and np.isfinite(y).all()
    for h in (1,4,8,16):
     e=pr[:,h-1]-y[:,h-1]
     er=dict(factual=float(np.mean(e**2)),effect=float(np.mean((response@e)**2)),interaction=float(np.mean((interaction@e)**2)))
     key=f'{int(g["shifted"])}/{n}/{h}'
     sample.setdefault(key,{}).setdefault(a,{})[id]=er
 for k,g in sample.items():
  cells[k]={a:{m:float(np.sqrt(np.mean([v[m] for v in g[a].values()]))) for m in ('factual','effect','interaction')} for a in ARMS}
 q=sample['1/16/8'];ids=sorted(groups.keys()&q['predictive-adaptive'].keys());assert len(ids)==32
 rng=np.random.default_rng(61063); draws=rng.integers(len(ids),size=(5000,len(ids))); comps={}
 for a in ('pca-adaptive','narx-adaptive','predictive-frozen'):
  ac=np.array([q['predictive-adaptive'][i]['effect'] for i in ids]); ba=np.array([q[a][i]['effect'] for i in ids])
  dd=np.sqrt(ac[draws].mean(1))-np.sqrt(ba[draws].mean(1))
  comps[a]={'delta':float(np.sqrt(ac.mean())-np.sqrt(ba.mean())),'ci':np.quantile(dd,[.025,.975]).tolist()}
 train=read_view(Path('train.json')); probes={}
 for a in ('predictive-adaptive','pca-adaptive','narx-adaptive'):
  with np.load(a.split('-')[0]+'.npz',allow_pickle=False) as m:
   xs=[];ys=[]
   for i,(o,u) in train.items():
    z=truth['calibration'][i]
    for j in range(4,len(o),4):xs.append(encode(m,o[:j+1]));ys.append(z[j//4])
  xs=np.array(xs);ys=np.array(ys);a1=np.c_[np.ones(len(xs)),xs];w=ridge(a1,ys,2.)
  xt=np.array([held[a][i] for i in sorted(groups)]); yt=np.array([groups[i]['checks']['16']['z'] for i in sorted(groups)])
  residual=yt-np.c_[np.ones(len(xt)),xt]@w
  probes[a]={'per_factor_r2':(1-(residual**2).sum(0)/((yt-yt.mean(0))**2).sum(0)).tolist(),'encoded_state_correlation':np.corrcoef(xt.T).tolist()}
 adaptive=cells['1/16/8']['predictive-adaptive'];pca=cells['1/16/8']['pca-adaptive'];stat=cells['0/16/8']
 r2=probes['predictive-adaptive']['per_factor_r2']
 gate061=(min(r2)>0 and np.mean(r2)>np.mean(probes['pca-adaptive']['per_factor_r2']))
 gate063=all(comps[a]['ci'][1]<0 for a in ('pca-adaptive','narx-adaptive')) and comps['predictive-frozen']['delta']<0 and adaptive['interaction']<pca['interaction'] and stat['predictive-adaptive']['factual']<=1.05*stat['pca-adaptive']['factual']
 for f,v in [('scores',{'scope':'CONSTRUCTED_NOT_QUALIFIED','cells':cells,'effect_comparisons':comps,'factor_probes':probes,'061_gate':bool(gate061),'063_gate':bool(gate063)}),('group-effects',sample),('metrics',{'groups':len(groups)})]:
  (out/(f+'.json')).write_text(json.dumps(v,allow_nan=False))
 write({"complete":len(cells),"groups":len(groups),"model_size":max(len(x) for a in held.values() for x in a.values()),"work_seconds":time.perf_counter()-started})
if __name__=='__main__':main()
