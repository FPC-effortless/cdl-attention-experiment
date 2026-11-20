"""Optimized/parity-aware pure CPU request processing, no learner fitting."""
import numpy as np
from gateway import Authority,admissible,select,serialize
ARMS=('slope','raw','pca','predictive','no-advice')
def prepare(model):
 v={}
 for arm,m in model.items():
  if arm=='slope':v[arm]=m;continue
  b,mu,sd=[np.array(m[k]) for k in ('beta','mean','std')]
  w=np.array(m['w']) if 'w' in m else None
  fv=(w@b if w is not None else b)/sd;fo=float(m['intercept']-mu@fv)
  v[arm]=(w,b,mu,sd,float(m['intercept']),fv,fo)
 return v
def observe(row):
 ev=[Authority(k,s,t,n,v) for k,s,t,n,v in row['events']]
 return admissible(ev,row['scope'],row['tick'],row['histories'])
def reference_scores(m,x):
 w,b,mu,sd,y,*_=m
 return ((x-mu)/sd)@w@b+y
def run(row,models,arm,fused=False):
 active=observe(row)
 if not active or arm=='no-advice':return serialize(None,active),False
 keys=tuple(active)
 x=np.asarray([active[k] for k in keys],dtype=float).reshape(len(keys),48)
 if not np.isfinite(x).all():raise ValueError('malformed sensor window')
 fallback=False
 if arm=='slope':
  j=models[arm]['sensor'];s=models[arm]['weight'];score=x[:,32+j]+s*(x[:,32+j]-x[:,16+j])
 elif fused:
  m=models[arm];score=x@m[-2]+m[-1]
  if len(keys)>1:
   pair=np.sort(score)[-2:]
   if pair[-1]-pair[-2] <= 1e-9:
    score=reference_scores(m,x);fallback=True
 elif arm=='raw':score=x@models[arm][-2]+models[arm][-1]
 else:score=reference_scores(models[arm],x)
 return serialize(select({k:float(s) for k,s in zip(keys,score)},active),active),fallback

def run_fused_group(rows,models,packed=False):
 """CPU only. Both implementations use equivalent standard affine folding."""
 if not packed:
  result=[run(r,models,'predictive',True) for r in rows]
  return [x[0] for x in result],sum(x[1] for x in result)
 admitted=[observe(r) for r in rows]
 refs=[(tuple(d),len(d)) for d in admitted]
 active=[d[k] for d,(keys,n) in zip(admitted,refs) for k in keys]
 if not active:return [serialize(None,d) for d in admitted],0
 x=np.asarray(active,dtype=float).reshape(len(active),48)
 if not np.isfinite(x).all():raise ValueError('malformed sensor window')
 m=models['predictive']; flat=x@m[-2]+m[-1]
 out=[];off=0;fallbacks=0
 for d,(keys,n) in zip(admitted,refs):
  v=flat[off:off+n];x0=x[off:off+n];off+=n
  if n>1:
   p=np.sort(v)[-2:]
   if p[-1]-p[-2]<=1e-9:v=reference_scores(m,x0);fallbacks+=1
  out.append(serialize(select({k:float(s) for k,s in zip(keys,v)},d),d))
 return out,fallbacks
