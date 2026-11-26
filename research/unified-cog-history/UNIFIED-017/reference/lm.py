import math,time,json,argparse,pathlib,torch
from torch import nn
import torch.nn.functional as F
from scan_reference import scan
P=pathlib.Path(__file__).parent
ARMS={'u1':(1,192,1,False,192,400),'u4':(4,48,1,False,192,600),'u8':(8,24,1,False,192,400),'u16':(16,12,1,False,192,400),'u32':(32,6,1,False,192,400),'wide8':(8,48,1,False,192,400),'ctg4':(4,48,1,True,192,400),'cold4':(4,48,.5,False,192,600),'integrated':(8,24,.5,True,192,400),'large4':(4,80,1,False,320,400),'largecold4':(4,80,.5,False,320,400)}
def rot(v,ph):
 a,b=v.unflatten(-1,(-1,2)).unbind(-1);c,s=ph.cos(),ph.sin();return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
def phase_scan(v,a,s,omega):
 if s is None:s=torch.zeros_like(v[:,0])
 ys=[]
 for z,w in zip(v.split(16,1),a.split(16,1)):
  ph=torch.arange(1,z.shape[1]+1,dtype=z.dtype,device=z.device)[:,None]*omega[None,:];y,_=scan(rot(z,-ph),w,s);y=rot(y,ph);s=y[:,-1];ys.append(y)
 return torch.cat(ys,1),s
class Block(nn.Module):
 def __init__(self,k,h,tau,ctg,d,ff):
  super().__init__();self.k=k;self.h=h;self.tau=tau;self.ctg=ctg;self.d=d;m=k*h;self.norm=nn.LayerNorm(d);self.inp=nn.Linear(d,2*d);self.pub=nn.Linear(d,m);self.route=nn.Linear(d,2*k+1);self.select=nn.Linear(d,2*k*4);self.ow=nn.Parameter(torch.empty(4,h,h));self.ob=nn.Parameter(torch.zeros(4,h));nn.init.uniform_(self.ow,-1/math.sqrt(h),1/math.sqrt(h));self.msgnorm=nn.LayerNorm(2*m);self.recv=nn.Linear(2*m,d);self.combine=nn.Linear(d,d);self.out=nn.Linear(d,d);self.norm2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d));self.register_buffer('omega',torch.logspace(-2,0,d//2),persistent=False)
 def op(self,x,p):return ((torch.einsum('...gh,koh->...gko',x,self.ow)+self.ob).tanh()*p[...,None]).sum(-2)
 def forward(self,h,state=None,ab=None):
  k,d=self.k,self.d;u=self.norm(h);z,a=self.inp(u).chunk(2,-1);a=.5+.49*a.sigmoid();a=a.unflatten(-1,(d//2,2)).mean(-1,keepdim=True).expand(*a.shape[:-1],d//2,2).flatten(-2)
  r,rs=phase_scan(z.tanh(),a,None if state is None else state[0],self.omega) if self.ctg else scan(z.tanh(),a,None if state is None else state[0]);q=r+u
  v=self.pub(q).tanh().unflatten(-1,(k,self.h));w,rp,g=self.route(q).split([k,k,1],-1);w=w.sigmoid();rp=rp.sigmoid();gate=.4*g.sigmoid()*w
  if ab=='uniform':gate=gate.mean(-1,keepdim=True).expand_as(gate);rp=rp.mean(-1,keepdim=True).expand_as(rp)
  if ab=='permute_read':rp=rp.roll(1,-1)
  ms,last=scan(v,1-gate[...,None],None if state is None else state[1]);m=ms*rp[...,None]*2
  p1,p2=(self.select(q).unflatten(-1,(2,k,4))/self.tau).softmax(-1).unbind(-3)
  y1=self.op(m,p1);y2=self.op(y1,p2);msg=self.recv(self.msgnorm(torch.cat([y1.flatten(-2),y2.flatten(-2)],-1)))
  if ab=='drop':msg=torch.zeros_like(msg)
  h=h+self.out(r+self.combine(q).sigmoid()*msg);h=h+self.ff(self.norm2(h));return h,(rs,last)
class LM(nn.Module):
 def __init__(self,arm,ff=None):
  super().__init__();k,h,tau,ctg,d,steps=ARMS[arm];self.arm=arm;self.d=d;target=1000000 if d==192 else 2800000
  if ff is None:
   b=Block(k,h,tau,ctg,d,1);base=2*sum(p.numel() for p in b.parameters())+256*d+2*d;ff=max(16,round(1+(target-base)/(2*(2*d+1))))
  self.ff=ff;self.emb=nn.Embedding(256,d);self.layers=nn.ModuleList([Block(k,h,tau,ctg,d,ff) for _ in range(2)]);self.norm=nn.LayerNorm(d);nn.init.normal_(self.emb.weight,std=.02)
 def forward(self,x,state=None,with_state=False,ab=None):
  h=self.emb(x);ss=[]
  for i,l in enumerate(self.layers):h,s=l(h,None if state is None else state[i],ab);ss.append(s)
  o=F.linear(self.norm(h),self.emb.weight);return (o,ss) if with_state else o
def data():
 ds={k:[torch.tensor(list(d['text'].encode()),dtype=torch.long) for d in v] for k,v in json.load(open(P/'corpus.json')).items()};tr=[bytes(d.tolist()) for d in ds['train']];ev={};excluded={}
 for k in ('dev','test'):
  rows=[];exc=0
  for d in ds[k]:
   if len(d)<65:continue
   for j in torch.linspace(0,len(d)-65,min(12,(len(d)-1)//64)).long().unique():
    a=d[j:j+65]
    if any(bytes(a.tolist()) in b for b in tr):exc+=1;continue
    rows.append(a)
  ev[k]=torch.stack(rows);excluded[k]=exc
 t=torch.cat(ds['train']);starts=[];off=0
 for d in ds['train']:starts.append(torch.arange(off,off+max(0,len(d)-64)));off+=len(d)
 return ds,t,torch.cat(starts),ev,excluded
@torch.no_grad()
def ev(m,rows,ab=None):
 m.eval();loss=0;ok=n=0
 for a in rows.split(16):
  z=m(a[:,:-1],ab=ab);y=a[:,1:];loss+=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum').item();ok+=(z.argmax(-1)==y).sum().item();n+=y.numel()
 return {'bpb':loss/n/math.log(2),'accuracy':ok/n,'n':n}
@torch.no_grad()
def hist(m,docs):
 losses=[0.,0.,0.];n=0
 for d in docs:
  if len(d)<1025:continue
  for j in torch.linspace(0,len(d)-1025,2).long().unique():
   a=d[j:j+1025][None];y=a[:,-64:]
   for i,l in enumerate([64,256,1024]):
    z=m(a[:,-l-1:-1])[:,-64:];losses[i]+=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum').item()
   n+=64
 return {'n':n,'bpb':[v/max(n,1)/math.log(2) for v in losses]}
@torch.no_grad()
def verify(m,d):
 m.eval();x=d[:128][None];z,s=m(x,with_state=True);a,st=m(x[:,:61],with_state=True);b,_=m(x[:,61:],state=st,with_state=True);y=torch.cat([a,b],1);st=None;ys=[]
 for q in x.split(1,1):o,st=m(q,state=st,with_state=True);ys.append(o)
 zz=torch.cat(ys,1);return {'n':x.numel(),'split_max':(z-y).abs().max().item(),'token_max':(z-zz).abs().max().item(),'argmax_disagree':(z.argmax(-1)!=zz.argmax(-1)).sum().item(),'state_bytes':sum(v.numel()*v.element_size() for l in s for v in l)}
@torch.no_grad()
def bench(m,d):
 m.eval();x=d[:128][None];_,s=m(x,with_state=True);q=x[:,:1];ts=[]
 for _ in range(3):
  st=s;t=time.perf_counter()
  for j in range(32):_,st=m(q,state=st,with_state=True)
  ts.append((time.perf_counter()-t)/32)
 return {'median_step_ms':sorted(ts)[1]*1000}
def run(arm,seed):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(seed);m=LM(arm);ds,t,starts,rows,excluded=data();steps=ARMS[arm][-1];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);g=torch.Generator().manual_seed(100000+seed);curve=[];sec=0.
 for i in range(1,steps+1):
  m.train();a=t[starts[torch.randint(len(starts),(8,),generator=g)][:,None]+torch.arange(65)];x,y=a[:,:-1],a[:,1:];opt.param_groups[0]['lr']=.001*min(1,i/50)*(.2+.8*.5*(1+math.cos(math.pi*i/600)));start=time.perf_counter();opt.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten());loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();sec+=time.perf_counter()-start
  if i in [250,400,600]:
   v=ev(m,rows['dev']);curve.append({'step':i,'dev':v});print(arm,seed,i,v,flush=True)
   if i==400 and steps>400:torch.save(m.state_dict(),P/f'lm_{arm}_{seed}_400.pt')
 key=f'lm_{arm}_{seed}';torch.save({'weights':m.state_dict(),'arm':arm,'seed':seed,'ff':m.ff},P/f'{key}.pt');r={'arm':arm,'seed':seed,'params':sum(p.numel() for p in m.parameters()),'ff':m.ff,'config':ARMS[arm],'steps':steps,'bytes':steps*512,'train_seconds':sec,'curve':curve,'test':ev(m,rows['test']),'dev':ev(m,rows['dev']),'history':hist(m,ds['test']),'stream':verify(m,max(ds['test'],key=len)),'timing':bench(m,max(ds['test'],key=len)),'excluded':excluded,'interventions':{ab:ev(m,rows['test'],ab) for ab in ['uniform','permute_read','drop']}}
 if steps>400:
  m.load_state_dict(torch.load(P/f'{key}_400.pt',weights_only=True));r['test400']=ev(m,rows['test'])
 (P/f'{key}.json').write_text(json.dumps(r,indent=2));print('DONE',key,r['test'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('arm',choices=list(ARMS));p.add_argument('seed',type=int);a=p.parse_args();run(a.arm,a.seed)
