"""019 experimental conditional parser. Supplied typed executor; unvalidated."""
import torch
from torch import nn
import lexer,executor,conditional

ARMS=('local_read','state_read','local_exact')
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__()
  if arm not in ARMS:raise ValueError(arm)
  self.arm=arm;self.tied=True;self.conditioned=arm=='state_read';self.is_gru=False;self.guarded=True;self.clause=True
  self.embedding=nn.Embedding(258,32)
  if self.is_gru:self.gru=nn.GRUCell(66,64);self.head=nn.Linear(64,106)
  else:
   self.frontend=nn.Linear(49*32+2,32);width=32 if self.conditioned else 48;self.head=nn.Sequential(nn.Linear(76 if self.conditioned else 32,width),nn.Tanh(),nn.Linear(width,113))
  if arm!='local_exact':self.gate_network=nn.Sequential(nn.Linear(9,16),nn.Tanh(),nn.Linear(16,2))
 def init(self,b):return executor.init(b),torch.zeros(b,64) if self.is_gru else torch.zeros(b,47,32),lexer.init(b),torch.zeros(b,dtype=torch.long),torch.ones(b,dtype=torch.bool)
 def decode(self,h,candidates):
  op,value,expected,gate,dp,sp=h.split([7,4,4,2,48,48],-1);anyid=(candidates>=0).any(-1)
  # Raw opcode logits remain unmasked for training and diagnostics.
  roles=[]
  for scores in (dp,sp):
   logits=torch.stack([torch.where(candidates==i,scores,torch.full_like(scores,-float('inf'))).max(-1).values for i in range(4)],-1)
   roles.append(torch.where(anyid[...,None],logits,torch.zeros_like(logits)))
  return torch.cat((op,*roles,value,expected,gate),-1)
 def segment(self,x,state,trace=False,execute=True,cache_features=True,return_raw=False,return_details=False,gold_events=None):
  mem,hist,lexstate,age,reset=state
  if self.conditioned and not execute:raise ValueError('owned predicted state required')
  resolved,current,candidates,lexnext,tables=lexer.scan(x,lexstate,tied=True)
  raw=self.embedding(resolved);look=self.embedding(current);ls=[];rs=[];ans=[];kh=[];dh=[];gate_targets=[]
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
   # Guard commit events to punctuation, then conditional decision to typed executor.
   allowed=torch.zeros_like(r[:,:7],dtype=torch.bool);allowed[:,0]=True
   allowed[:,1:5]=(x[:,t]==46)[:,None];allowed[:,6]=x[:,t]==46;allowed[:,5]=x[:,t]==63
   predicted_src=r[:,11:15].argmax(-1);predicted_expected=r[:,19:23].argmax(-1)+1
   source_value=executor.read(mem,predicted_src) if self.arm!='local_exact' or gold_events is not None else torch.zeros_like(predicted_src)
   if self.arm!='local_exact':
    from torch.nn import functional as F
    gate_logits=self.gate_network(torch.cat((F.one_hot(source_value,5),F.one_hot(predicted_expected-1,4)),-1).float())
    r=torch.cat((r[:,:23],gate_logits),-1)
   effective_op=r[:,:7].masked_fill(~allowed,-float('inf'))
   noop=torch.full_like(effective_op,-float('inf'));noop[:,0]=0
   effective_op=torch.where((cand>=0).any(-1)[:,None],effective_op,noop)
   l=torch.cat((effective_op,r[:,7:]),-1);rs.append(r);ls.append(l)
   op,dst,src,val,expected,gate=(z.argmax(-1) for z in l.split([7,4,4,4,4,2],-1));val=val+1;expected=expected+1
   # Target only: gold event position selects auxiliary target; predicate uses OWN predicted operands and state, never gold roles/values.
   gt=torch.full_like(op,-100)
   if gold_events is not None:
    for row,e in enumerate(gold_events):
     q=e.get(t)
     if q is not None and q['op']==6:
      gt[row]=(source_value[row]==predicted_expected[row]).long()
   gate_targets.append(gt)
   if execute and bool((op!=0).any()):mem,a=conditional.apply(mem,op,dst,src,val,expected,None if self.arm=='local_exact' else gate.bool())
   else:a=torch.full_like(op,-1)
   ans.append(a)
   if trace:kh.append(mem[0]);dh.append(mem[1])
   age=age+1;reset=x[:,t]==10
  result=(torch.stack(rs if return_raw else ls,1),torch.stack(ans,1),(mem,hist,lexnext,age,reset),(torch.stack(kh,1),torch.stack(dh,1)) if trace else None)
  return (*result,torch.stack(rs,1),torch.stack(gate_targets,1)) if return_details else result

def tensor(cases):return torch.tensor([list(q['text'].encode()) for q in cases])
