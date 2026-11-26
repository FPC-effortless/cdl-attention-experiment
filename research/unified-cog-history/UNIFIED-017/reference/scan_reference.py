import argparse, json, math, pathlib, hashlib, time, statistics, os
import torch
from torch import nn
import torch.nn.functional as F
P=pathlib.Path(__file__).parent
BASE=pathlib.Path('/opt/pyvenv/lib/python3.13/site-packages/gensim/test/test_data')
K=4;D=196;V=256;T=64
# Affine recurrence in short chunks; a>=.2. No absolute positional embedding.
def scan(z,a,s=None):
 if s is None:s=torch.zeros_like(z[:,0])
 if z.size(1)==1:
  s=a[:,0]*s+(1-a[:,0])*z[:,0];return s[:,None],s
 ys=[]
 for zc,ac in zip(z.split(16,1),a.split(16,1)):
  p=ac.cumprod(1);y=p*((1-ac)*zc/p).cumsum(1)+p*s[:,None];s=y[:,-1];ys.append(y)
 return torch.cat(ys,1),s
class Block(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.norm=nn.LayerNorm(D);self.inp=nn.Linear(D,2*D);self.pub=nn.Linear(D,D);self.route=nn.Linear(D,2*K+1);self.recv=nn.Linear(D,D);self.combine=nn.Linear(D,D);self.out=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,635),nn.GELU(),nn.Linear(635,D))
 def forward(self,h,state=None,ab=None):
  u=self.norm(h);z,a=self.inp(u).chunk(2,-1);r,rs=scan(z.tanh(),.5+.49*a.sigmoid(),None if state is None else state[0]);q=r+u;v=self.pub(q).tanh().unflatten(-1,(K,D//K));w,rp,g=self.route(q).split([K,K,1],-1)
  if self.arm=='independent':w=w.sigmoid()*2/K;rp=rp.sigmoid()*2/K
  else:w=w.softmax(-1);rp=rp.softmax(-1)
  if ab=='uniform':w=torch.full_like(w,1/K);rp=torch.full_like(rp,1/K)
  if ab=='swap_write':w=w.roll(1,-1)
  if ab=='swap_read':rp=rp.roll(1,-1)
  gate=.8*g.sigmoid()*w
  ms,mslast=scan(v,1-gate[...,None],None if state is None else state[1]);m=ms*rp[...,None]*K
  if isinstance(ab,int):m=m*F.one_hot(torch.tensor(ab,device=m.device),K).logical_not().to(m.dtype)[None,None,:,None]
  if ab=='drop':m=torch.zeros_like(m)
  msg=self.recv(m.flatten(-2));h=h+self.out(r+self.combine(q).sigmoid()*msg);h=h+self.ff(self.norm2(h));return h,(rs,mslast),(w,rp)
class LM(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.emb=nn.Embedding(V,D);self.layers=nn.ModuleList([Block(arm) for _ in range(2)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,V,bias=False);self.head.weight=self.emb.weight;nn.init.normal_(self.emb.weight,std=.02)
 def forward(self,x,state=None,ab=None,with_aux=False):
  h=self.emb(x);ss=[];ds=[]
  for i,l in enumerate(self.layers):h,s,di=l(h,None if state is None else state[i],ab);ss.append(s);ds.append(di)
  out=self.head(self.norm(h));return (out,ss,ds) if with_aux else out

def get_docs():
 seen=set();groups=[];raw=(BASE/'lee_background.cor').read_bytes().splitlines(keepends=True);fresh=(BASE/'lee.cor').read_bytes().splitlines(keepends=True)
 for xs in [raw[:240],raw[240:270],raw[270:],fresh]:
  ds=[]
  for x in xs:
   if x.strip() in seen:continue
   seen.add(x.strip());ds.append(x)
  groups.append(ds)
 return groups

def make_train(docs):
 t=torch.tensor(list(b''.join(docs)),dtype=torch.long);offset=0;starts=[]
 for x in docs:
  starts.extend(range(offset,offset+max(0,len(x)-T)));offset+=len(x)
 return t,torch.tensor(starts)

def rows(docs,T):
 out=[]
 for d in docs:
  t=torch.tensor(list(d),dtype=torch.long)
  for j in range(0,len(t)-T,T):out.append(t[j:j+T+1])
 return torch.stack(out)

@torch.no_grad()
def evaluate(m,ds,ab=None,route_stats=False):
 m.eval();loss=0;ok=n=0;counts=torch.zeros(2,2,K);mass=torch.zeros_like(counts);ent=torch.zeros(2,2);individual=[]
 for doc in ds:
  if len(doc)<=T:continue
  dl=0;dn=0
  for a in rows([doc],T).split(16):
   x,y=a[:,:-1],a[:,1:];z,_,di=m(x,ab=ab,with_aux=True);l=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum').item();loss+=l;dl+=l;dn+=y.numel();n+=y.numel();ok+=(z.argmax(-1)==y).sum().item()
   if route_stats:
    for i,(w,rp) in enumerate(di):
     for j,g in enumerate((w,rp)):
      g=g/g.sum(-1,keepdim=True);counts[i,j]+=F.one_hot(g.argmax(-1),K).sum((0,1));mass[i,j]+=g.sum((0,1));ent[i,j]+=-(g*g.clamp_min(1e-10).log()).sum().item()
  individual.append([dl/math.log(2),dn])
 res=dict(bpb=loss/n/math.log(2),acc=ok/n,n=n,docs=len(individual),doc_loss=individual)
 if route_stats:
  probs=mass/n;res.update(top_counts=counts.tolist(),mean_route=probs.tolist(),mean_token_entropy=(ent/n/math.log(K)).tolist(),mutual_info_nats=(-(probs*probs.clamp_min(1e-10).log()).sum(-1)-ent/n).tolist())
 return res

def regularizer(ds):
 terms=[]
 for pair in ds:
  for p in pair:
   mean=p.mean((0,1));terms.append(math.log(K)+(mean*mean.clamp_min(1e-10).log()).sum()-(p*p.clamp_min(1e-10).log()).sum(-1).mean())
 return torch.stack(terms).mean()

@torch.no_grad()
def history(m,docs):
 m.eval();loss=[0.,0.];n=0
 for doc in docs:
  t=torch.tensor(list(doc),dtype=torch.long)
  for j in range(0,len(t)-256,256):
   a=t[j:j+257][None];y=a[:,193:257];z=m(a[:,:256])[:,-64:];short=m(a[:,192:256]);
   for k,o in enumerate([short,z]):loss[k]+=F.cross_entropy(o.flatten(0,1),y.flatten(),reduction='sum').item()
   n+=y.numel()
 return {'n':n,'reset_bpb':loss[0]/n/math.log(2),'full_bpb':loss[1]/n/math.log(2)}

@torch.no_grad()
def streaming(m,doc):
 m.eval();x=torch.tensor(list(doc[:512]),dtype=torch.long)[None];z,s,_=m(x,with_aux=True);st=None;ys=[]
 for i in range(x.size(1)):
  o,st,_=m(x[:,i:i+1],state=st,with_aux=True);ys.append(o)
 out=torch.cat(ys,1);return dict(n=x.size(1),max_logit_diff=(z-out).abs().max().item(),argmax_disagree=(z.argmax(-1)!=out.argmax(-1)).sum().item(),state_bytes=sum(v.numel()*v.element_size() for layer in s for v in layer),causal_max_diff=(z[:,:128]-m(x[:,:128])).abs().max().item())

def train(a):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);tr,dev,old,fresh=get_docs();t,starts=make_train(tr);m=LM(a.arm);nparams=sum(p.numel() for p in m.parameters());g=torch.Generator().manual_seed(100000+a.seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);log=[];elapsed=0.;m.train()
 for i in range(1,a.steps+1):
  pos=starts[torch.randint(len(starts),(a.batch,),generator=g)];data=t[pos[:,None]+torch.arange(T+1)[None]];x,y=data[:,:-1],data[:,1:];s=time.perf_counter();opt.param_groups[0]['lr']=.001*min(1,i/50)*(.2+.8*.5*(1+math.cos(math.pi*i/a.steps)));opt.zero_grad(set_to_none=True);z,_,di=m(x,with_aux=True);ce=F.cross_entropy(z.flatten(0,1),y.flatten());reg=regularizer(di) if a.arm=='balanced' else torch.tensor(0.);loss=ce+.05*reg;loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();elapsed+=time.perf_counter()-s
  if i in [1,a.steps//2,a.steps]:
   val=evaluate(m,dev);m.train();log.append(dict(step=i,ce=ce.item(),reg=reg.item(),val_bpb=val['bpb']));print(a.arm,a.seed,i,log[-1],flush=True)
 ck=P/f'{a.arm}_{a.seed}.pt';torch.save(dict(arm=a.arm,seed=a.seed,weights=m.state_dict()),ck)
 res=dict(arm=a.arm,seed=a.seed,params=nparams,steps=a.steps,batch=a.batch,context=T,train_bytes=a.steps*a.batch*T,train_seconds=elapsed,history_train=log,dev=evaluate(m,dev),old=evaluate(m,old),fresh=evaluate(m,fresh,route_stats=True),history=history(m,fresh),stream=streaming(m,max(fresh,key=len)))
 res['interventions']={str(ab):evaluate(m,fresh,ab=ab) for ab in ['uniform','swap_write','swap_read','drop',0,1,2,3]};(P/f'{a.arm}_{a.seed}.json').write_text(json.dumps(res,indent=2));print('DONE',a.arm,a.seed,nparams,res['fresh']['bpb'],res['fresh']['top_counts'],res['history'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['independent','competitive','balanced'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch',type=int,default=8);train(p.parse_args())
