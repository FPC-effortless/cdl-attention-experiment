"""017 NEW candidate, derived from archived 014. Not a recovered 016 model."""
import torch
from torch import nn
import lexer,executor
from guard import guard_torch
ARMS=('context_mlp','guard_context_mlp','guard_clause_mlp','guard_clause_state','guard_clause_gru')
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__()
  if arm not in ARMS:raise ValueError(arm)
  self.arm=arm;self.tied=True;self.conditioned=arm.endswith('state');self.is_gru=arm.endswith('gru');self.guarded=arm.startswith('guard');self.clause='clause' in arm
  self.embedding=nn.Embedding(258,32)
  if self.is_gru:self.gru=nn.GRUCell(66,64);self.head=nn.Linear(64,106)
  else:
   self.frontend=nn.Linear(49*32+2,32);width=32 if self.conditioned else 48;self.head=nn.Sequential(nn.Linear(76 if self.conditioned else 32,width),nn.Tanh(),nn.Linear(width,106))
 def init(self,b):return executor.init(b),torch.zeros(b,64) if self.is_gru else torch.zeros(b,47,32),lexer.init(b),torch.zeros(b,dtype=torch.long),torch.ones(b,dtype=torch.bool)
 def decode(self,h,candidates):
  op,value,dp,sp=h.split([6,4,48,48],-1);anyid=(candidates>=0).any(-1)
  # Raw opcode logits remain unmasked for training and diagnostics.
  roles=[]
  for scores in (dp,sp):
   logits=torch.stack([torch.where(candidates==i,scores,torch.full_like(scores,-float('inf'))).max(-1).values for i in range(4)],-1)
   roles.append(torch.where(anyid[...,None],logits,torch.zeros_like(logits)))
  return torch.cat((op,*roles,value),-1)
 def segment(self,x,state,trace=False,execute=True,cache_features=True,return_raw=False,return_details=False):
  mem,hist,lexstate,age,reset=state
  if self.conditioned and not execute:raise ValueError('owned predicted state required')
  resolved,current,candidates,lexnext,tables=lexer.scan(x,lexstate,tied=True)
  raw=self.embedding(resolved);look=self.embedding(current);ls=[];rs=[];ans=[];kh=[];dh=[]
  for t in range(x.shape[1]):
   age=torch.where(reset,torch.zeros_like(age),age)
   if self.clause:hist=torch.where(reset.reshape((-1,)+(1,)*(hist.ndim-1)),torch.zeros_like(hist),hist)
   terminal=(x[:,t]==46)|(x[:,t]==63);bf=torch.stack((terminal.float(),age.clamp(max=48).float()/48),-1)
   # Resolved token is delayed one byte. At first byte of a clause it belongs to the previous clause.
   emb=torch.where((reset & self.clause)[:,None],torch.zeros_like(raw[:,t]),raw[:,t])
   if self.is_gru:hist=self.gru(torch.cat((emb,look[:,t],bf),-1),hist);front=hist
   else:
    window=torch.cat((hist,emb[:,None]),1);front=self.frontend(torch.cat((window.flatten(1),look[:,t],bf),-1)).tanh();hist=window[:,1:]
   cand=candidates[:,t]
   if self.clause:
    distance=torch.arange(47,-1,-1,device=x.device)[None,:]
    cand=torch.where(distance<age[:,None],cand,torch.full_like(cand,-1))
   features=executor.features(mem) if self.conditioned else None
   r=self.decode(self.head(torch.cat((front,features),-1) if self.conditioned else front),cand)
   effective_op=guard_torch(r[:,:6],x[:,t],self.guarded)
   noop=torch.full_like(effective_op,-float('inf'));noop[:,0]=0
   effective_op=torch.where((cand>=0).any(-1)[:,None],effective_op,noop)
   l=torch.cat((effective_op,r[:,6:]),-1);rs.append(r);ls.append(l)
   op,dst,src,val=(z.argmax(-1) for z in l.split([6,4,4,4],-1));val=val+1
   if execute and bool((op!=0).any()):mem,a=executor.step(mem,op,dst,src,val)
   else:a=torch.full_like(op,-1)
   ans.append(a)
   if trace:kh.append(mem[0]);dh.append(mem[1])
   age=age+1;reset=x[:,t]==10
  result=(torch.stack(rs if return_raw else ls,1),torch.stack(ans,1),(mem,hist,lexnext,age,reset),(torch.stack(kh,1),torch.stack(dh,1)) if trace else None)
  return (*result,torch.stack(rs,1)) if return_details else result

def tensor(cases):return torch.tensor([list(q['text'].encode()) for q in cases])
