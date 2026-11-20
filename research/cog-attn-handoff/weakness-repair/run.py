import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, torch, json, time, bisect
from torch import nn
from pathlib import Path
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
N=8; A=2*N*N
S,D=np.indices((N,N));src=np.tile(S.ravel(),2);dst=np.tile(D.ravel(),2);op=np.repeat(np.arange(2),N*N)
def data(seed,n,alias,cohort):
 r=np.random.default_rng(seed);s=[];a=[];m=[];v=[];states=[];ys=[];gold=[]
 for i in range(n):
  while True:
   x,y=r.choice(N,2,replace=False)
   if ((x+y)%4==0)==(cohort=='held_pair'):break
  al=[int(r.choice(np.flatnonzero(alias==j))) for j in (x,y)]
  style=int(r.integers(2)); vs=1-style if cohort=='recombine' else style
  o=int(r.integers(2));verb=o*2+vs
  if cohort=='novel':verb=4+o
  mark=[2*style,2*style+1]
  if r.integers(2):al=al[::-1];mark=mark[::-1]
  st=r.integers(0,4,N);target=st.copy();target[y]=st[x] if o==0 else (st[x]+st[y])%4
  result=np.tile(st,(A,1));result[np.arange(A),dst]=np.where(op==0,st[src],(st[src]+st[dst])%4)
  mask=np.all(result==target,axis=1)
  a.append(al);m.append(mark);v.append(verb);states.append(st);ys.append(target);gold.append(o*N*N+x*N+y);s.append(mask)
 return tuple(torch.tensor(np.array(z)) for z in (a,m,v,states,ys,gold,s))
class Factor(nn.Module):
 def __init__(self):
  super().__init__();self.alias=nn.Embedding(16,N);self.role=nn.Embedding(4,1);self.verb=nn.Embedding(6,2)
  for x in self.parameters():nn.init.normal_(x,std=.1)
 def forward(self,a,m,v):
  e=self.alias(a).softmax(-1);p=(self.role(m[:,0])-self.role(m[:,1])).sigmoid().view(-1,1,1)
  z=p*e[:,0,:,None]*e[:,1,None,:]+(1-p)*e[:,1,:,None]*e[:,0,None,:]
  return (self.verb(v).softmax(-1)[:,:,None,None]*z[:,None]).reshape(-1,A).clamp_min(1e-12).log()
class Flat(nn.Module):
 def __init__(self):
  super().__init__();self.e=nn.Embedding(16,16);self.m=nn.Embedding(4,8);self.v=nn.Embedding(6,8);self.net=nn.Sequential(nn.Linear(56,64),nn.ReLU(),nn.Linear(64,A))
 def forward(self,a,m,v):return self.net(torch.cat((self.e(a).flatten(1),self.m(m).flatten(1),self.v(v)),1)).log_softmax(-1)
def eval_model(model,d):
 a,m,v,st,y,g,mask=d
 with torch.no_grad():
  log=model(a,m,v);pick=log.argmax(1);ok=mask[torch.arange(len(g)),pick];conf=log.exp().max(1).values
 return dict(n=len(g),correct=int(ok.sum()),action_correct=int((pick==g).sum()),highconf=int((conf>=.95).sum()),highconf_correct=int((ok&(conf>=.95)).sum()))
def train_all():
 rows=[]
 for seed in [11,29,47]:
  r=np.random.default_rng(seed);alias=r.permutation(np.repeat(np.arange(N),2));tr=data(seed+1000,4096,alias,'familiar');tests={c:data(seed+2000+i,1024,alias,c) for i,c in enumerate(['familiar','held_pair','recombine','novel'])}
  for mode in ['factor_soft','factor_hard','factor_gold','flat_soft']:
   torch.manual_seed(seed);model=Flat() if mode=='flat_soft' else Factor();opt=torch.optim.Adam(model.parameters(),lr=.03);gen=torch.Generator().manual_seed(seed);t=time.perf_counter()
   for j in range(400):
    i=torch.randint(len(tr[0]),(64,),generator=gen);a,m,v,st,y,g,mask=[z[i] for z in tr];lp=model(a,m,v)
    if mode.endswith('soft'):loss=-lp.masked_fill(~mask,-1e9).logsumexp(1).mean()
    else:
     target=g if mode=='factor_gold' else mask.int().argmax(1);loss=-lp[torch.arange(len(i)),target].mean()
    opt.zero_grad();loss.backward();opt.step()
   row=dict(seed=seed,mode=mode,params=sum(p.numel() for p in model.parameters()),train_s=time.perf_counter()-t,cohorts={c:eval_model(model,d) for c,d in tests.items()});rows.append(row);torch.save(dict(state=model.state_dict(),alias=alias.tolist()),P/f'{mode}_{seed}.pt');print(seed,mode,{c:v['correct'] for c,v in row['cohorts'].items()},flush=True)
 return rows

def scan(events,k,t,cut):
 best=(-1,-1,-1)
 for e,vt,tx,val in events:
  if e==k and vt<=t and tx<=cut and (vt,tx)>best[:2]:best=(vt,tx,val)
 return best[2]
def lookup(index,k,t,cut):
 a=index[k];i=bisect.bisect_right(a,(t,10**9,10**9))-1
 while i>=0:
  if a[i][1]<=cut:return a[i][2]
  i-=1
 return -1

def history(seed,T=4096,Q=256):
 r=np.random.default_rng(seed);events=[];query=[];index=[[] for _ in range(32)];latest=[(-1,-1,-1)]*32
 for tx in range(T):
  k=int(r.integers(32));vt=tx if r.random()>.15 else max(0,tx-int(r.integers(1,256)));v=int(r.integers(1000));events.append((k,vt,tx,v))
 for _ in range(Q):
  k=int(r.integers(32));t=int(r.integers(T));cut=int(r.integers(t,T));query.append((k,t,cut))
 p=time.perf_counter_ns()
 for k,vt,tx,v in events:
  bisect.insort(index[k],(vt,tx,v));latest[k]=max(latest[k],(vt,tx,v))
 prep=time.perf_counter_ns()-p;p=time.perf_counter_ns();ref=[scan(events,*q) for q in query];sc=time.perf_counter_ns()-p;p=time.perf_counter_ns();out=[lookup(index,*q) for q in query];ix=time.perf_counter_ns()-p
 l=[latest[k][2] for k,t,cut in query];current=[(k,T-1,T-1) for k in range(32)];assert [lookup(index,*q) for q in current]==[scan(events,*q) for q in current]==[x[2] for x in latest]
 return dict(seed=seed,T=T,n=Q,exact=sum(a==b for a,b in zip(ref,out)),latest_correct=sum(a==b for a,b in zip(ref,l)),current=32,scan_ns=sc,index_ns=ix,prep_ns=prep,log_payload=T*32,index_payload=T*24+32*24)
if __name__=='__main__':
 rows=train_all();h=[history(s) for s in list(range(100,132))+list(range(300,332))];(P/'results.json').write_text(json.dumps(dict(neural=rows,history=h),indent=2));print('history',sum(x['exact'] for x in h),sum(x['latest_correct'] for x in h),flush=True)
