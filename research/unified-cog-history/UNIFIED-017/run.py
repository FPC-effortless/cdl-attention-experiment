import os, sys, math, json, time, hashlib, pathlib, random, concurrent.futures
import torch
from torch import nn
import torch.nn.functional as F
P=pathlib.Path(__file__).parent
sys.path.insert(0,str(P/'reference'))
from lm import Block
from scan_check import assoc
import lm
lm.scan=assoc
SEEDS=[5209,5231,5279]; STEPS=300; BATCH=12; TTRAIN=16; N=8; V=8; D=32; VOCAB=34
# IDs 0..7 identity roots; 8..11 alias modifiers; 12..19 values.
FOR,NOT,SURE,MAYBE,SET,CLEAR,KEEP,QUERY=range(20,28)
ORDERS=[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def batch(seed,b,t,panel='iid',delayed=False):
 g=torch.Generator().manual_seed(seed)
 k=torch.randint(N,(b,t),generator=g); val=torch.randint(V,(b,t),generator=g); verb=torch.multinomial(torch.tensor([.55,.25,.20]),b*t,replacement=True,generator=g).view(b,t); amb=torch.rand(b,t,generator=g)<.2; target=torch.randint(N,(b,),generator=g)
 if delayed:
  k=(target[:,None]+torch.randint(1,N,(b,t),generator=g))%N;k[:,0]=target;verb[:,0]=0;amb[:,0]=False
 d=(k+torch.randint(1,N,(b,t),generator=g))%N
 heldalias=panel in ('alias','both'); heldorder=panel in ('order','both')
 def mod(ids):
  # withheld single modifier per identity; different from entirely unseen lexical token
  h=(3-ids)%4
  return h if heldalias else (h+torch.randint(1,4,ids.shape,generator=g))%4
 a=verb+SET; action=torch.where(amb,3,verb)
 c=torch.stack([torch.where(amb,MAYBE,SURE),a,val+12],-1)
 s=torch.stack([torch.full_like(k,FOR),mod(k)+8,k],-1)
 n=torch.stack([torch.full_like(d,NOT),mod(d)+8,d],-1)
 spans=torch.stack([c,s,n],-2)
 orders=torch.tensor(ORDERS);ix=torch.randint(4,(b,t),generator=g) if not heldorder else torch.randint(4,6,(b,t),generator=g)
 x=spans.gather(-2,orders[ix][...,None].expand(b,t,3,3)).flatten(-2)
 # explicit canonical queries, no label supplied for destination at inference
 y=torch.full((b,N),8,dtype=torch.long);ii=torch.arange(b)
 for j in range(t):
  prev=y[ii,k[:,j]];y[ii,k[:,j]]=torch.where(action[:,j]==0,val[:,j],torch.where(action[:,j]==1,8,prev))
 return x,action,k,val,y,target

class COG(nn.Module):
 def __init__(self,ff):super().__init__();self.b=Block(4,D//4,1.,False,D,ff)
 def forward(self,x):return self.b(x)[0]
class Attn(nn.Module):
 def __init__(self,ff):
  super().__init__();self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.out=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,ff),nn.GELU(),nn.Linear(ff,D))
 def forward(self,x):
  t=x.shape[1];freq=torch.exp(-math.log(10000)*torch.arange(D//2,device=x.device)/ (D//2));theta=torch.arange(t,device=x.device)[:,None]*freq;pos=torch.stack([theta.sin(),theta.cos()],-1).flatten(-2);x=x+pos[None]*.1
  q,k,v=self.qkv(self.norm(x)).reshape(x.shape[0],t,3,4,D//4).permute(2,0,3,1,4).unbind(0)
  h=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape_as(x);x=x+self.out(h);return x+self.ff(self.norm2(x))
class GRU(nn.Module):
 def __init__(self,ff):super().__init__();self.norm=nn.LayerNorm(D);self.rnn=nn.GRU(D,D,batch_first=True);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,ff),nn.GELU(),nn.Linear(ff,D))
 def forward(self,x):x=x+self.rnn(self.norm(x))[0];return x+self.ff(self.norm2(x))
class Delta(nn.Module):
 # Pedagogical gated delta-rule recurrence; not official Gated DeltaNet.
 def __init__(self,ff):super().__init__();self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.gates=nn.Linear(D,8);self.out=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,ff),nn.GELU(),nn.Linear(ff,D))
 def forward(self,x):
  b,t,_=x.shape;z=self.norm(x);q,k,v=self.qkv(z).reshape(b,t,3,4,8).unbind(2);q=F.normalize(q,dim=-1);k=F.normalize(k,dim=-1);a,beta=self.gates(z).sigmoid().reshape(b,t,2,4).unbind(2);a=.5+.49*a;s=torch.zeros(b,4,8,8,device=x.device,dtype=x.dtype);ys=[]
  for i in range(t):
   s=s*a[:,i,:,None,None];ki=k[:,i];vi=v[:,i];err=vi-torch.einsum('bhk,bhkv->bhv',ki,s);s=s+beta[:,i,:,None,None]*ki.unsqueeze(-1)*err.unsqueeze(-2);ys.append(torch.einsum('bhk,bhkv->bhv',q[:,i],s).flatten(1))
  x=x+self.out(torch.stack(ys,1));return x+self.ff(self.norm2(x))
CLASSES={'cog':COG,'transformer':Attn,'gru':GRU,'delta':Delta}
class Net(nn.Module):
 def __init__(self,arm,ff):
  super().__init__();self.arm=arm;self.ff=ff;self.emb=nn.Embedding(VOCAB,D);self.event=CLASSES[arm](ff);self.history=CLASSES[arm](ff);self.en=nn.LayerNorm(D);self.hn=nn.LayerNorm(D);self.action=nn.Linear(D,4);self.key=nn.Linear(D,N);self.value=nn.Linear(D,V);self.query=nn.Embedding(N,D);self.read=nn.Linear(D,V+1)
  with torch.no_grad():
   for m in self.modules():
    if isinstance(m,nn.Linear):m.weight.normal_(0,1/math.sqrt(m.in_features));m.bias.zero_() if m.bias is not None else None
   nn.init.normal_(self.emb.weight,std=.1);nn.init.normal_(self.query.weight,std=.1)
   for m in self.modules():
    if isinstance(m,Block):m.ow.normal_(0,1/math.sqrt(m.h))
 def encode(self,x):
  b,t,l=x.shape;e=self.emb(x).reshape(b*t,l,D);h=self.en(self.event(e).mean(1)).reshape(b,t,D);return h,(self.action(h),self.key(h),self.value(h))
 def forward(self,x):
  h,heads=self.encode(x);q=self.query.weight[None].expand(h.shape[0],-1,-1);z=self.history(torch.cat([h,q],1));return self.read(self.hn(z[:,-N:])),heads

def npar(m):return sum(p.numel() for p in m.parameters())
def ffsize(arm):
 base=npar(Net(arm,1));slope=npar(Net(arm,2))-base;return max(4,round(1+(24000-base)/slope))

def decode(heads):
 pp=[h.softmax(-1) for h in heads];(ap,kp,vp)=[p.argmax(-1) for p in pp];conf=pp[0].amax(-1)*pp[1].amax(-1)*torch.where(ap==0,pp[2].amax(-1),torch.ones_like(pp[2].amax(-1)));return ap,kp,vp,conf

def event_counts(pred,truth):
 a,k,v,c=pred;ta,tk,tv=truth;w=ta==0;relevant=ta<2;j=(a==ta)&(~relevant | ((k==tk)&(~w|(v==tv))));return dict(n=a.numel(),joint=int(j.sum()),action=int((a==ta).sum()),key=int((k==tk).sum()),writes=int(w.sum()),payload=int(((v==tv)&w).sum()),legit=int(relevant.sum()))

def commits(pred,truth,threshold):
 a,k,v,c=pred;ta,tk,tv=truth;accept=(a<2)&(c>=threshold);legit=ta<2;right=accept&(a==ta)&(k==tk)&((ta!=0)|(v==tv));return dict(accepted=int(accept.sum()),right=int(right.sum()),unsafe=int((accept&~right).sum()),legit=int(legit.sum()),missed=int((legit&~right).sum()),n=a.numel())

def tensor_store(pred,threshold=0,initial=None,check=False):
 a,k,v,c=pred;b,t=a.shape;s=torch.full((b,N),8,dtype=torch.long) if initial is None else initial.clone();ii=torch.arange(b);checks=bad=updates=0
 for j in range(t):
  do=(a[:,j]<2)&(c[:,j]>=threshold);old=s.clone() if check else None;val=torch.where(a[:,j]==0,v[:,j],8);s[ii,k[:,j]]=torch.where(do,val,s[ii,k[:,j]])
  if check:
   mask=torch.ones_like(s,dtype=torch.bool);mask[ii,k[:,j]]=~do;checks+=int(mask.sum());bad+=int((s[mask]!=old[mask]).sum());updates+=int(do.sum())
 return s,(checks,bad,updates)

def dict_store(pred,threshold=0,initial=None):
 a,k,v,c=[x.tolist() for x in pred];ds=[{} for _ in a] if initial is None else [dict(d) for d in initial]
 for i in range(len(a)):
  for aa,kk,vv,cc in zip(a[i],k[i],v[i],c[i]):
   if cc>=threshold:
    if aa==0:ds[i][kk]=vv
    elif aa==1:ds[i].pop(kk,None)
 return torch.tensor([[d.get(k,8) for k in range(N)] for d in ds]),ds

def slice_pred(p,sl):return tuple(x[:,sl] for x in p)
def merge(a,b):
 for k,v in b.items():a[k]=a.get(k,0)+v

def score(z,y,target):
 ok=z==y;return dict(n_hist=len(y),complete=int(ok.all(1).sum()),individual=int(ok.sum()),n_answers=y.numel(),target=int(ok[torch.arange(len(y)),target].sum()))

@torch.no_grad()
def calibration(m,seed):
 m.eval();ts=[0,.5,.7,.85,.95,.99];out={str(t):{} for t in ts}
 for i in range(16):
  x,a,k,v,_,_=batch(8000000+seed*100+i,16,16);_,heads=m.encode(x);p=decode(heads)
  for t in ts:merge(out[str(t)],commits(p,(a,k,v),t))
 valid=[t for t in ts if out[str(t)]['accepted'] and out[str(t)]['unsafe']/out[str(t)]['accepted']<=.01]
 return (max(valid,key=lambda t:(out[str(t)]['right'],-t)) if valid else 1.),out

@torch.no_grad()
def evaluate(m,seed,threshold,quick=False):
 m.eval();out={}
 # fresh seeds distinct from training and calibration; every panel same per model within seed
 for pi,panel in enumerate(['iid','order','alias','both']):
  for t in [16,128,1024]:
   num=8 if t<1024 else 4;b=8 if t<1024 else 4
   e={};raw={};gated={};native={};base={};gated_s={};oracle={};checks=bad=updates=parity=external=0
   for i in range(num):
    x,a,k,v,y,target=batch(9000000+seed*1000+pi*10000+t*10+i,b,t,panel,delayed=t>16)
    z,heads=m(x);p=decode(heads);merge(e,event_counts(p,(a,k,v)));merge(raw,commits(p,(a,k,v),0));merge(gated,commits(p,(a,k,v),threshold));merge(native,score(z.argmax(-1),y,target));s,cs=tensor_store(p,check=True);ss,dd=dict_store(p);external+=int((s!=ss).sum());checks+=cs[0];bad+=cs[1];updates+=cs[2]
    g,_=tensor_store(p,threshold);merge(base,score(s,y,target));merge(gated_s,score(g,y,target));half=t//2;first,_=tensor_store(slice_pred(p,slice(0,half)));buf=first.numpy().tobytes();loaded=torch.frombuffer(bytearray(buf),dtype=torch.int64).reshape_as(first);res,_=tensor_store(slice_pred(p,slice(half,None)),initial=loaded);parity+=int((res!=s).sum());o,_=tensor_store((a,k,v,torch.ones_like(a,dtype=torch.float)));merge(oracle,score(o,y,target))
   out[f'{panel}_{t}']=dict(event=e,raw_commit=raw,calibrated_commit=gated,native=native,store=base,calibrated_store=gated_s,oracle=oracle,checks=checks,bad=bad,updates=updates,parity=parity,external=external)
 return out

def run(arm,seed,regime):
 torch.set_num_threads(1);torch.manual_seed(seed);ff=ffsize(arm);torch.manual_seed(seed);m=Net(arm,ff);opt=torch.optim.AdamW(m.parameters(),lr=.003);curve=[];t=time.perf_counter()
 for i in range(STEPS):
  m.train();x,a,k,v,y,_=batch(100000+seed*1000+i,BATCH,TTRAIN);z,h=m(x);la,lk,lv=h;loss=F.cross_entropy(z.flatten(0,1),y.flatten());
  if regime=='joint':loss=loss+(F.cross_entropy(la.flatten(0,1),a.flatten())+F.cross_entropy(lk.flatten(0,1),k.flatten())+F.cross_entropy(lv.flatten(0,1),v.flatten()))/3
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();assert torch.isfinite(loss)
  if (i+1)%100==0:curve.append(dict(step=i+1,loss=float(loss.detach()),native=float((z.argmax(-1)==y).float().mean()),event=event_counts(decode(h),(a,k,v))));print('STEP',arm,seed,regime,curve[-1],flush=True)
 train_seconds=time.perf_counter()-t;path=P/'fits'/f'{arm}_{regime}_{seed}.pt';torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),arm=arm,ff=ff,seed=seed,regime=regime,steps=STEPS),path)
 threshold,dev=calibration(m,seed);ev=evaluate(m,seed,threshold);r=dict(arm=arm,seed=seed,regime=regime,ff=ff,params=npar(m),steps=STEPS,histories=STEPS*BATCH,events=STEPS*BATCH*TTRAIN,train_seconds=train_seconds,curve=curve,threshold=threshold,calibration=dev,eval=ev,sha256=hashlib.sha256(path.read_bytes()).hexdigest());path.with_suffix('.json').write_text(json.dumps(r,indent=2));print('DONE',arm,regime,seed,r['params'],threshold,{k:(z['event']['joint'],z['event']['n'],z['store']['complete'],z['store']['n_hist']) for k,z in ev.items() if k in ('iid_16','order_16','both_1024')},flush=True)
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 jobs=[(a,s,'joint') for a in CLASSES for s in SEEDS]+[(a,s,'endpoint') for a in ('cog','transformer') for s in SEEDS];random.Random(17001).shuffle(jobs)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  fs={pool.submit(run,*j):j for j in jobs}
  for f in concurrent.futures.as_completed(fs):
   try:f.result()
   except Exception as e:print('FAIL',fs[f],repr(e),flush=True);raise
