import torch,math,json,pathlib,time,hashlib
from torch import nn
import torch.nn.functional as F
from hierarchical import hierarchical
from hold_reset import exact_hold_reset
P=pathlib.Path(__file__).parent; SEEDS=(173,271,907); ARMS=('soft','hard','anneal','aux','gru'); K=8; V=8; H=16; D=32
COL=['amber','blue','green','red']; AN=['fox','owl','hare','wolf']; VAL=['silver','gold','copper','iron','tin','jade','pearl','ruby']
TRAIN=[['a b was near v','v was near a b','a b mentioned v'],['a b now holds v','record v for a b','a b receives v'],['clear a b not v','remove v from a b','a b no longer holds v']]
HELD=[['near a b was v','v mentioned a b'],['v now a b holds','for a b record v'],['a b clear not v','from a b remove v']]
WORDS=['<pad>','<unk>']+COL+AN+VAL+sorted(set(w for grp in TRAIN+HELD for t in grp for w in t.split() if w not in ['a','b','v'])); W={w:i for i,w in enumerate(WORDS)}; MAX=8
NAME=torch.tensor([[W[a],W[b]] for a in COL for b in AN]); KNOWN=torch.tensor([i for i in range(16) if i//4!=i%4]); UNKNOWN=torch.tensor([i for i in range(16) if i//4==i%4]); VALUES=torch.tensor([W[x] for x in VAL]);
def table(ts):
 t=torch.zeros(3,max(len(x) for x in ts),16,8,2,MAX,dtype=torch.long)
 for a,grp in enumerate(ts):
  for j,pat in enumerate(grp):
   for k in range(16):
    for v in range(8):
     for alias in range(2):
      n=NAME[k].flip(0) if alias else NAME[k];row=[int(n[0]) if x=='a' else int(n[1]) if x=='b' else int(VALUES[v]) if x=='v' else W[x] for x in pat.split()];t[a,j,k,v,alias,:len(row)]=torch.tensor(row)
 return t
TT,HH=table(TRAIN),table(HELD)
def gen(g,n,L,kind='mixed'):
 if kind=='unknown':
  known=KNOWN[torch.rand(n,12,generator=g).argsort(1)[:,:4]];roster=torch.cat([known,UNKNOWN.expand(n,4)],1);roster=roster.gather(1,torch.rand(n,K,generator=g).argsort(1))
 else:roster=KNOWN[torch.rand(n,12,generator=g).argsort(1)[:,:K]]
 addr=torch.randint(K,(n,L),generator=g);v=torch.randint(V,(n,L),generator=g);u=torch.rand(n,L,generator=g);a=torch.where(u<.5,0,torch.where(u<.85,1,2));alias=(torch.rand(n,L,generator=g)<.25).long()
 if kind=='alias':alias.fill_(1)
 if kind in ('retention','reset'):
  addr[:,:K]=torch.arange(K);a[:,:K]=1;a[:,K:]=0
  if kind=='reset':addr[:,K:K+4]=torch.rand(n,K,generator=g).argsort(1)[:,:4];a[:,K:K+4]=2
 if kind=='conflict':a[:]=1;addr[:,:2]=addr[:,2:3];v[:,1]=(v[:,0]+1)%V
 j=torch.arange(n);y=torch.zeros(n,K,dtype=torch.long)
 for i in range(L):
  z=addr[:,i];y[j,z]=torch.where(a[:,i]==1,v[:,i]+1,torch.where(a[:,i]==2,0,y[j,z]))
 tab=HH if kind=='paraphrase' else TT;tid=torch.randint(tab.shape[1],(n,L),generator=g);entity=roster.gather(1,addr);words=tab[a,tid,entity,v,alias];v2=torch.where(a!=1,(v+1)%V,v);cf=tab[a,tid,entity,v2,alias]
 return words,NAME[roster],a,addr,v,y,cf
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(len(WORDS),D,padding_idx=0);self.pos=nn.Parameter(torch.randn(MAX,D)*.05);self.mix=nn.Linear(D,D);self.act=nn.Sequential(nn.Linear(D,D),nn.GELU(),nn.Linear(D,3));self.val=nn.Linear(D,V);self.q=nn.Linear(D,D,bias=False);self.name=nn.Linear(D,D,bias=False)
 def forward(self,w,names):
  mask=(w!=0)[...,None];h=self.emb(w);z=F.gelu(self.mix(h+self.pos))*mask;pooled=z.sum(-2)/mask.sum(-2).clamp_min(1);nam=self.name(self.emb(names).mean(-2));q=self.q(pooled);addr=torch.einsum('btd,bkd->btk',q,nam)/math.sqrt(D);return pooled,self.act(pooled),addr,self.val(pooled)
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.enc=Encoder();self.value=nn.Parameter(torch.randn(V,H)*.4);self.read=nn.Sequential(nn.Linear(H,32),nn.GELU(),nn.Linear(32,V+1));self.gru=nn.GRU(D+K,K*H,batch_first=True) if arm=='gru' else None
 def forward(self,w,names,state=None,mode=None):
  p,al,kl,vl=self.enc(w,names);mode=mode or ('hard' if self.arm in ('hard','anneal') else 'soft');a,k,v=al.softmax(-1),kl.softmax(-1),vl.softmax(-1)
  if self.gru is not None:
   yy,last=self.gru(torch.cat([p,k],-1),None if state is None else state.flatten(1).unsqueeze(0));s=last.squeeze(0).reshape(-1,K,H);return self.read(s),s,(al,kl,vl),yy.reshape(w.shape[0],w.shape[1],K,H)
  if mode=='hard':
   def st(x):
    h=F.one_hot(x.argmax(-1),x.shape[-1]).float();return h+(x-x.detach()) if self.training else h
   a,k,v=map(st,(a,k,v))
  value=v@self.value;mass=a[...,1:2]+a[...,2:3];ret=1-k*mass;pro=(value*a[...,1:2]/mass.clamp_min(1e-9))[...,None,:].expand(-1,-1,K,-1)
  if mode=='hard' and not self.training:pro=torch.where((a[...,2:3]>0)[...,None],torch.zeros_like(pro),pro)
  if state is None:state=torch.zeros_like(pro[:,0])
  yy,s=exact_hold_reset(pro,ret[...,None],state) if mode=='hard' else hierarchical(pro,ret[...,None],state,accum=torch.float32)
  return self.read(s),s,(al,kl,vl),yy
PANELS=[('mixed',16),('mixed',64),('mixed',256),('mixed',1024),('retention',1024),('reset',1024),('conflict',256),('unknown',64),('paraphrase',64),('alias',64)]
@torch.no_grad()
def evaluate(m,seed,mode=None,panels=PANELS,n=128):
 m.eval();out={}
 for kind,L in panels:
  g=torch.Generator().manual_seed(1409000+seed*31+L*37+['mixed','retention','reset','conflict','unknown','paraphrase','alias'].index(kind));r=dict(n=n,complete=0,slots=0,cf_both=0,bitwise=0,split=0,action=0,address=0,value=0,event_n=0,write_n=0)
  for _ in range(n//16):
   w,names,a,k,v,y,cf=gen(g,16,L,kind);z,s,ls,_=m(w,names,mode=mode);zz,ss,_,_=m(cf,names,mode=mode);ok=(z.argmax(-1)==y).all(1);ok2=(zz.argmax(-1)==y).all(1);r['complete']+=int(ok.sum());r['slots']+=int((z.argmax(-1)==y).sum());r['cf_both']+=int((ok&ok2).sum());r['bitwise']+=int((s.contiguous().view(torch.int32)==ss.contiguous().view(torch.int32)).flatten(1).all(1).sum());_,st,_,_=m(w[:,:L//2],names,mode=mode);end,_,_,_=m(w[:,L//2:],names,st,mode=mode);r['split']+=int((z.argmax(-1)==end.argmax(-1)).all(1).sum());r['action']+=int((ls[0].argmax(-1)==a).sum());r['address']+=int((ls[1].argmax(-1)==k).sum());r['value']+=int(((ls[2].argmax(-1)==v)&(a==1)).sum());r['event_n']+=a.numel();r['write_n']+=int((a==1).sum())
  out[f'{kind}_{L}']=r
 return out

def train(arm,seed,steps=400):
 torch.set_num_threads(1);torch.manual_seed(seed);m=Model(arm);g=torch.Generator().manual_seed(1401000+seed);opt=torch.optim.AdamW(m.parameters(),lr=.004);start=time.perf_counter();curve=[]
 for i in range(1,steps+1):
  m.train();w,n,a,k,v,y,_=gen(g,32,16);mode='soft' if arm in ('soft','aux') or (arm=='anneal' and i<=300) else 'hard';z,s,ls,_=m(w,n,mode=mode);loss=F.cross_entropy(z.flatten(0,1),y.flatten())
  if arm=='aux':loss+=.3*(F.cross_entropy(ls[0].flatten(0,1),a.flatten())+F.cross_entropy(ls[1].flatten(0,1),k.flatten())+F.cross_entropy(ls[2][a==1],v[a==1]))
  opt.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step()
  if i in [1,100,200,300,400]:curve.append(dict(step=i,loss=float(loss.detach()),complete=float((z.argmax(-1)==y).all(1).float().mean())))
 key=f'{arm}_{seed}';path=P/'fits'/f'{key}.pt';torch.save(dict(weights=m.state_dict(),arm=arm,seed=seed,steps=steps,vocabulary=WORDS),path);r=dict(arm=arm,seed=seed,steps=steps,examples=steps*32,params=sum(p.numel() for p in m.parameters()),seconds=time.perf_counter()-start,curve=curve,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native=evaluate(m,seed));
 if arm in ('soft','aux'):r['hardened']=evaluate(m,seed,'hard')
 (P/'fits'/f'{key}.json').write_text(json.dumps(r,indent=2));print('DONE',key,[(k,v['complete'],v['bitwise']) for k,v in r['native'].items()],flush=True)
if __name__=='__main__':
 import concurrent.futures,random
 torch.set_num_threads(1);torch.set_num_interop_threads(1);jobs=[(a,s) for a in ARMS for s in SEEDS];random.Random(14003).shuffle(jobs)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for f in concurrent.futures.as_completed([pool.submit(train,*j) for j in jobs]):f.result()
