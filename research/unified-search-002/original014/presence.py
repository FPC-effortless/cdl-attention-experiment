import torch,math,json,time,hashlib,random,concurrent.futures
from torch import nn
import torch.nn.functional as F
import custody014 as c
from curriculum import Factor,curriculum
from custody014 import P,K,V,H,D,SEEDS
from hierarchical import hierarchical
from hold_reset import exact_hold_reset
ARMS=('presence','presence_curriculum')
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.enc=Factor();self.proj=nn.Linear(D,H);self.read=nn.Sequential(nn.Linear(H,32),nn.GELU(),nn.Linear(32,V))
 def forward(self,w,names,state=None,mode=None):
  p,al,kl,_=self.enc(w,names);a,k=al.softmax(-1),kl.softmax(-1);value=self.proj(p).tanh();vl=self.read(value);mode=mode or 'soft'
  if mode=='hard':
   def st(x):
    h=F.one_hot(x.argmax(-1),x.shape[-1]).float();return h+x-x.detach() if self.training else h
   a,k=st(a),st(k)
  mass=a[...,1:2]+a[...,2:3];ret=1-k*mass;presence=a[...,1:2]/mass.clamp_min(1e-9);pro=torch.cat([value*presence,presence],-1)[...,None,:].expand(-1,-1,K,-1)
  if mode=='hard' and not self.training:pro=torch.where((a[...,2:3]>0)[...,None],torch.zeros_like(pro),pro)
  if state is None:state=torch.zeros_like(pro[:,0])
  yy,s=exact_hold_reset(pro,ret[...,None],state) if mode=='hard' else hierarchical(pro,ret[...,None],state,accum=torch.float32)
  p=s[...,-1:].clamp(1e-7,1-1e-7);lp=torch.cat([(1-p).log(),p.log()+self.read(s[...,:H]).log_softmax(-1)],-1);return lp,s,(al,kl,vl),yy

def run(arm,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);m=Model(arm);g=torch.Generator().manual_seed(1421000+seed);opt=torch.optim.AdamW(m.parameters(),lr=.004);curve=[];events=0;t=time.perf_counter()
 for i in range(1,401):
  b=curriculum(g,32,2) if 'curriculum' in arm and i<=160 else c.gen(g,32,4) if 'curriculum' in arm and i<=240 else c.gen(g,32,16);w,n,a,k,v,y,_=b;events+=a.numel();m.train();z,_,_,_=m(w,n);loss=F.nll_loss(z.flatten(0,1),y.flatten());opt.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step()
  if i in (1,160,240,400):curve.append(dict(step=i,loss=float(loss.detach()),complete=float((z.argmax(-1)==y).all(1).float().mean())))
 key=f'{arm}_{seed}';path=P/'fits'/f'{key}.pt';torch.save(dict(weights=m.state_dict(),arm=arm,seed=seed,steps=400,vocabulary=c.WORDS),path);r=dict(arm=arm,seed=seed,steps=400,examples=12800,event_exposures=events,params=sum(p.numel() for p in m.parameters()),seconds=time.perf_counter()-t,curve=curve,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native=c.evaluate(m,seed),hardened=c.evaluate(m,seed,'hard'),fresh_hardened=c.evaluate(m,seed+10000,'hard',panels=[('mixed',1024),('retention',1024),('reset',1024),('unknown',64),('paraphrase',64),('alias',64)]));(P/'fits'/f'{key}.json').write_text(json.dumps(r,indent=2));print('DONE',key,[(k,v['complete'],v['bitwise']) for k,v in r['hardened'].items()],flush=True)
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);jobs=[(a,s) for a in ARMS for s in SEEDS];random.Random(14005).shuffle(jobs)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
  for f in concurrent.futures.as_completed([p.submit(run,*j) for j in jobs]):f.result()
