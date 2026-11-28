import math,time,json,hashlib,torch,random,concurrent.futures
from torch import nn
import torch.nn.functional as F
import custody014 as c
from custody014 import P,K,V,NAME,KNOWN,TT,SEEDS
ARMS=('curriculum','factor','factor_curriculum','factor_aux')
class Factor(c.Encoder):
 def forward(self,w,names):
  mask=(w!=0)[...,None];h=self.emb(w);z=F.gelu(self.mix(h+self.pos))*mask;pooled=z.sum(-2)/mask.sum(-2).clamp_min(1)
  # Shared lexical matching. Roster name words and sentence words are not parsed into IDs beyond tokenizer.
  x=F.normalize(h,dim=-1);q=F.normalize(self.emb(names),dim=-1);sim=torch.einsum('btwd,bknd->btknw',x,q);sim=sim.masked_fill((w==0)[:,:,None,None,:],-2.);addr=15*sim.max(-1).values.sum(-1)
  return pooled,self.act(pooled),addr,self.val(pooled)
class Model(c.Model):
 def __init__(self,arm):
  super().__init__('soft');self.variant=arm
  if arm.startswith('factor'):
   prev=self.enc;self.enc=Factor();self.enc.load_state_dict(prev.state_dict())
def curriculum(g,n,L):
 roster=KNOWN[torch.rand(n,12,generator=g).argsort(1)[:,:K]];addr=torch.randint(K,(n,L),generator=g);v=torch.randint(V,(n,L),generator=g);a=torch.randint(3,(n,L),generator=g);a[:,0]=1
 if L==2:addr[:,1]=torch.where(torch.rand(n,generator=g)<.5,addr[:,0],addr[:,1]);
 alias=(torch.rand(n,L,generator=g)<.25).long();j=torch.arange(n);y=torch.zeros(n,K,dtype=torch.long)
 for i in range(L):
  z=addr[:,i];y[j,z]=torch.where(a[:,i]==1,v[:,i]+1,torch.where(a[:,i]==2,0,y[j,z]))
 tid=torch.randint(3,(n,L),generator=g);ent=roster.gather(1,addr);return TT[a,tid,ent,v,alias],NAME[roster],a,addr,v,y,None

def run(arm,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);m=Model(arm);g=torch.Generator().manual_seed(1421000+seed);opt=torch.optim.AdamW(m.parameters(),lr=.004);curve=[];events=0;start=time.perf_counter()
 for i in range(1,401):
  b=curriculum(g,32,2) if 'curriculum' in arm and i<=160 else c.gen(g,32,4) if 'curriculum' in arm and i<=240 else c.gen(g,32,16);w,n,a,k,v,y,_=b;events+=a.numel();m.train();z,_,ls,_=m(w,n);loss=F.cross_entropy(z.flatten(0,1),y.flatten())
  if arm=='factor_aux':loss+=.3*(F.cross_entropy(ls[0].flatten(0,1),a.flatten())+F.cross_entropy(ls[1].flatten(0,1),k.flatten())+F.cross_entropy(ls[2][a==1],v[a==1]))
  opt.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step()
  if i in (1,160,240,400):curve.append(dict(step=i,loss=float(loss.detach()),complete=float((z.argmax(-1)==y).all(1).float().mean())))
 key=f'{arm}_{seed}';path=P/'fits'/f'{key}.pt';torch.save(dict(weights=m.state_dict(),arm=arm,seed=seed,steps=400,vocabulary=c.WORDS),path);r=dict(arm=arm,seed=seed,steps=400,examples=12800,event_exposures=events,params=sum(p.numel() for p in m.parameters()),seconds=time.perf_counter()-start,curve=curve,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native=c.evaluate(m,seed),hardened=c.evaluate(m,seed,'hard'),fresh_hardened=c.evaluate(m,seed+10000,'hard',panels=[('mixed',1024),('retention',1024),('reset',1024),('unknown',64),('paraphrase',64),('alias',64)]));(P/'fits'/f'{key}.json').write_text(json.dumps(r,indent=2));print('DONE',key,[(k,v['complete'],v['bitwise']) for k,v in r['hardened'].items()],flush=True)
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);jobs=[(a,s) for a in ARMS for s in SEEDS];random.Random(14004).shuffle(jobs)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
  for f in concurrent.futures.as_completed([p.submit(run,*j) for j in jobs]):f.result()
