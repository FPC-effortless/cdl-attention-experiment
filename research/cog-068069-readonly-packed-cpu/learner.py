"""Compact predictive/PCA and raw history fit; no simulator or held truth import."""
import json
from pathlib import Path
import numpy as np
def load(p):return json.loads(Path(p).read_text())
def check_view(rows,v,fields):
 return {r['id']:{k:r[k] for k in fields} for r in rows}=={r['id']:r['features'] for r in v['rows']} and len(rows)==len(v['rows'])
def mat(data,sensor):
 xs=[];ys=[]
 for u in data:
  for stream in u['streams']:
   a=np.array(stream,dtype=float)
   h=np.concatenate((a[:-3],a[1:-2],a[2:-1]),axis=1)
   xs.append(h);ys.append(a[3:,sensor])
 return np.concatenate(xs),np.concatenate(ys)
def ridge(x,y):
 return np.linalg.solve(x.T@x+2*np.eye(x.shape[1]),x.T@y)
def fit(train,cal,sensor,threshold):
 x,yt=mat(train,sensor); c,cy=mat(cal,sensor)
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-6); y0=yt.mean();x=(x-mu)/sd
 # Supervisory index y[t+1] is shared. Predictive projection uses next
 # observed 16-vector rather than the single target label; same sensor stream,
 # extra unsupervised prediction objective disclosed. PCA sees equal histories.
 nxt=np.concatenate([np.array(a)[3:] for r in train for a in r['streams']])
 root=np.linalg.cholesky(x.T@x+4*np.eye(x.shape[1]))
 whiten=np.linalg.solve(root,x.T).T
 proj=np.linalg.solve(root.T,np.linalg.svd(whiten.T@(nxt-nxt.mean(0)),full_matrices=False)[0][:,:4])*np.sqrt(len(x))
 pca=np.linalg.svd(x,full_matrices=False)[2][:4].T
 coef={}
 for arm,w in [('raw',np.eye(48)),('pca',pca),('predictive',proj)]:
  w=w/np.maximum((x@w).std(0),1e-6)
  c1=ridge(x@w,yt-y0);coef[arm]={'w':w.tolist(),'beta':c1.tolist(),'mean':mu.tolist(),'std':sd.tolist(),'intercept':float(y0)}
 # Cheap analytic last-plus-slope tuned only on fixed CAL.
 # Ranking objective uses threshold classification balanced by target disparity.
 bins=[0.,.25,.5,1.,1.5]
 def margin(s):return float(np.mean((c[:,32+sensor]+s*(c[:,32+sensor]-c[:,16+sensor])-threshold-(cy-threshold))**2))
 del coef['raw']['w']
 coef['slope']={'weight':min(bins,key=lambda v:(margin(v),v)),'sensor':sensor}
 return coef,{'fit_rows':len(x),'cal_rows':len(c),'pred_trainable_coefficient_storage':48*4+4+2*48+1,'raw_effective_coefficients':48+1,'fitted_arms':3}
