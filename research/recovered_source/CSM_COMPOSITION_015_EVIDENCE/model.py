"""Learned mention roles over engineered causal identity interning."""
import torch
from torch import nn
import lexer,executor
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.tied=arm.startswith('tied');self.conditioned=arm.endswith('state');self.is_gru=arm.endswith('gru');self.embedding=nn.Embedding(258,32)
  if self.is_gru:self.gru=nn.GRU(64,64,batch_first=True);self.head=nn.Linear(64,106)
  else:
   self.frontend=nn.Linear(49*32,32);width=32 if self.conditioned else 48;self.head=nn.Sequential(nn.Linear(76 if self.conditioned else 32,width),nn.Tanh(),nn.Linear(width,106))
 def init(self,b):return executor.init(b), (torch.zeros(1,b,64) if self.is_gru else torch.zeros(b,47,32)),lexer.init(b)
 def decode(self,h,candidates):
  op,value,dp,sp=h.split([6,4,48,48],-1);anyid=(candidates>=0).any(-1)
  # No identity candidate means no admissible action, explicitly shared by controls.
  op=torch.cat((op[...,:1],torch.where(anyid[...,None],op[...,1:],torch.full_like(op[...,1:],-1e9))),-1)
  roles=[]
  for scores in (dp,sp):
   logits=torch.stack([torch.where(candidates==i,scores,torch.full_like(scores,-1e9)).max(-1).values for i in range(4)],-1)
   logits=torch.where(anyid[...,None],logits,torch.zeros_like(logits));roles.append(logits)
  return torch.cat((op,*roles,value),-1)
 def segment(self,x,state,trace=False,execute=True,cache_features=True):
  mem,hist,lexstate=state;resolved,current,candidates,lexnext,tables=lexer.scan(x,lexstate,tied=self.tied);raw=self.embedding(resolved);look=self.embedding(current)
  if self.is_gru:front,newhist=self.gru(torch.cat((raw,look),-1),hist)
  else:
   context=torch.cat((hist,raw),1);windows=context.unfold(1,48,1).permute(0,1,3,2).flatten(2);front=self.frontend(torch.cat((windows,look),-1)).tanh();newhist=context[:,-47:]
  if not self.conditioned:
   logits=self.decode(self.head(front),candidates)
   if not execute:return logits,None,(mem,newhist,lexnext),None
  else:
   if not execute:raise ValueError('conditioned must own predicted state')
   features=executor.features(mem);ls=[]
  ans=[];kh=[];dh=[]
  for t in range(x.shape[1]):
   if self.conditioned:
    if not cache_features:features=executor.features(mem)
    l=self.decode(self.head(torch.cat((front[:,t],features),-1)),candidates[:,t]);ls.append(l)
   else:l=logits[:,t]
   op,dst,src,val=(z.argmax(-1) for z in l.split([6,4,4,4],-1));val=val+1
   if bool((op!=0).any()):
    mem,a=executor.step(mem,op,dst,src,val)
    if self.conditioned and bool(((op>=1)&(op<=4)).any()):features=executor.features(mem)
   else:a=torch.full_like(op,-1)
   ans.append(a)
   if trace:kh.append(mem[0]);dh.append(mem[1])
  if self.conditioned:logits=torch.stack(ls,1)
  return logits,torch.stack(ans,1),(mem,newhist,lexnext),(torch.stack(kh,1),torch.stack(dh,1)) if trace else None

def tensor(cases):return torch.tensor([list(q['text'].encode()) for q in cases])
