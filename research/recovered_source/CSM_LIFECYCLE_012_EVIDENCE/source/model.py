"""Learned action selection, common exact typed executor; no supplied action at inference."""
import torch
from torch import nn
import executor

def actions(logits):
 op,dst,src,val=torch.split(logits,[6,4,4,4],-1)
 return op.argmax(-1),dst.argmax(-1),src.argmax(-1),val.argmax(-1)+1
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.conditioned=arm=='state';self.embedding=nn.Embedding(256,32)
  if arm=='gru':self.gru=nn.GRU(32,64,batch_first=True);self.head=nn.Linear(64,18)
  else:
   self.frontend=nn.Linear(48*32,32);self.head=nn.Sequential(nn.Linear(76 if self.conditioned else 32,32 if self.conditioned else 64),nn.Tanh(),nn.Linear(32 if self.conditioned else 64,18))
 def init(self,b):return executor.init(b),(torch.zeros(1,b,64) if self.arm=='gru' else torch.zeros(b,47,32))
 def segment(self,x,state,trace=False,execute=True,cache_features=True):
  mem,history=state;raw=self.embedding(x)
  if self.arm=='gru':front,newhistory=self.gru(raw,history)
  else:
   context=torch.cat((history,raw),1);front=self.frontend(context.unfold(1,48,1).permute(0,1,3,2).flatten(2)).tanh();newhistory=context[:,-47:]
  if not self.conditioned:
   logits=self.head(front)
   if not execute:return logits,None,(mem,newhistory),None
  else:
   if not execute:raise ValueError('state-conditioned parser must execute its predicted actions')
   features=executor.features(mem);logitlist=[]
  ans=[];kh=[];dh=[]
  for t in range(x.shape[1]):
   if self.conditioned:
    if not cache_features:features=executor.features(mem)
    l=self.head(torch.cat((front[:,t],features),-1));logitlist.append(l)
   else:l=logits[:,t]
   op,dst,src,val=actions(l)
   if bool((op!=0).any()):
    mem,a=executor.step(mem,op,dst,src,val)
    if self.conditioned and bool(((op>=1)&(op<=4)).any()):features=executor.features(mem)
   else:a=torch.full_like(op,-1)
   ans.append(a)
   if trace:kh.append(mem[0]);dh.append(mem[1])
  if self.conditioned:logits=torch.stack(logitlist,1)
  return logits,torch.stack(ans,1),(mem,newhistory),(torch.stack(kh,1),torch.stack(dh,1)) if trace else None

def tensor(cases):return torch.tensor([list(q['text'].encode()) for q in cases])
def targets(cases):
 b=len(cases);t=len(cases[0]['text']);y=torch.full((b,t,4),-100,dtype=torch.long);y[:,:,0]=0
 for i,q in enumerate(cases):
  for e in q['events']:
   p=e['pos'];y[i,p,0]=e['op']
   if e['op']>0:y[i,p,1]=e['dst']
   if e['op'] in (2,3):y[i,p,2]=e['src']
   if e['op']==1:y[i,p,3]=e['value']-1
 return y

def loss(logits,y,weights):
 from torch.nn import functional as F
 pieces=torch.split(logits,[6,4,4,4],-1);ls=[]
 for i,p in enumerate(pieces):
  mask=y[:,:,i]!=-100
  ls.append(F.cross_entropy(p[mask],y[:,:,i][mask],weight=weights if i==0 else None) if bool(mask.any()) else p.sum()*0)
 return sum(ls)
