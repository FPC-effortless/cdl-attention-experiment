"""Constructed continuous data, made prior to model fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from forge_data.core import canonical
from forge_data.diagnostics import diagnose
D=16; Z=4; H=16; SHIFT=40; SUPPORT=(0,4,16)
TRAIN=range(1000,1064); DEV_SHIFT=range(2000,2032); DEV_STABLE=range(3000,3032)
ROOT=Path(__file__).resolve().parent

def mix():
 r=np.random.default_rng(61063)
 M=np.linalg.qr(r.normal(size=(D,7)))[0]
 return M[:,:Z], M[:,Z:],r.uniform(.17,.32,size=(Z,2))*np.array([[1,-1],[1,1],[-1,1],[1,-1]])

def episode(seed,n=128,shift=True):
 rng=np.random.default_rng(seed)
 w,nmat,B=mix()
 A=np.diag([.55,.48,.61,.47])+rng.normal(0,.032,(Z,Z));np.fill_diagonal(A,[.55,.48,.61,.47])
 A[1,0]+=.17;A[2,1]-=.19;A[3,2]+=.13
 drift=np.zeros((Z,Z));drift[1,0]=-.39;drift[2,1]=.36;drift[3,0]=.25
 u=rng.uniform(-.8,.8,(n,2)); e=rng.normal(0,.014,(n,Z)); q=rng.normal(0,.13,(n,3)); m=rng.normal(0,.014,(n+1,D))
 x=np.zeros((n+1,Z)); v=np.zeros((n+1,3));x[0]=rng.uniform(-.45,.45,Z);v[0]=rng.normal(0,.35,3)
 def step(s,t,act):
  a=A+(drift if shift and t>=SHIFT else 0)
  cross=np.array([.11*s[1]*s[2],.18*s[0]*act[0],.16*act[0]*act[1],-.12*s[0]*s[3]])
  return np.tanh(a@s+B@act+cross+e[t])
 for t in range(n):
  x[t+1]=step(x[t],t,u[t]);v[t+1]=.92*v[t]+q[t]
 obs=x@w.T+1.65*v@nmat.T+m
 def branch(t,act):
  y=x[t].copy(); out=[]
  for i,p in enumerate(act):
   y=step(y,t+i,p)
   out.append(y@w.T+1.65*v[t+i+1]@nmat.T+m[t+i+1])
  return np.array(out)
 return obs,u,x,branch

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,item):
 b=canonical(item)
 if p.exists() and p.read_bytes()!=b:raise ValueError(f'nonidentical frozen {p.name}')
 p.write_bytes(b)
def view(rows,split):
 return {'schema':'forge.diagnostic-input.v1','feature_fields':['o','u'],'group_fields':['group'],'rows':[{'id':str(k),'split':split,'features':{'o':o.tolist(),'u':u.tolist()},'groups':{'group':str(k)}} for k,o,u in rows]}
def main():
 d=ROOT/'frozen';d.mkdir(exist_ok=True); a=[];b=[];truth={'calibration':{},'evaluation':{}}
 for k in TRAIN:
  o,u,z,_=episode(k)
  a.append((k,o,u));truth['calibration'][str(k)]=z[::4].tolist()
 for k in [*DEV_SHIFT,*DEV_STABLE]:
  o,u,z,bran=episode(k,n=80,shift=k in DEV_SHIFT)
  b.append((k,o[:SHIFT+max(SUPPORT)+1],u[:SHIFT+max(SUPPORT)]))
  group={'shifted':k in DEV_SHIFT,'checks':{}}
  for n in SUPPORT:
   t=SHIFT+n; acts=[]
   for c0,c1 in [(-1,-1),(1,-1),(-1,1),(1,1)]:
    plan=np.zeros((H,2));plan[:4]=np.array([c0,c1])*.65;acts.append({'u':plan.tolist(),'y':bran(t,plan).tolist()})
   group['checks'][str(n)]={'branches':acts,'z':z[t].tolist()}
  truth['evaluation'][str(k)]=group
 for name,doc in [('train',view(a,'train')),('challenge',view(b,'validation'))]:
  p=d/(name+'.json');put(p,doc);r=diagnose(p,sha256=digest(p),max_rows=1000,max_findings=8);put(d/(name+'-forge.json'),r)
 put(d/'truth.json',truth)
 # Separate causal target from its provenance: score only, never a fit input.
 s={'train_groups':len(a),'dev_groups':len(b),'group_intersection':len(set(TRAIN)&set([*DEV_SHIFT,*DEV_STABLE])),'train_transitions':sum(len(u) for _,_,u in a),'rank':4,'sensor_dim':D,'lag':3,'ridge_fits_per_arm':1,'planned_threads':1,'max_seconds':900,'paired_controls':4}
 put(d/'counts.json',s)
 print(s)
if __name__=='__main__':main()
