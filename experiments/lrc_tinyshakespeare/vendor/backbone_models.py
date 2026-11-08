"""Established backbone/cache controls, not a novel LRC architecture claim."""
from collections import OrderedDict,deque
from types import SimpleNamespace
import math,sys
import torch
from torch import nn
from torch.nn import functional as F
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM,StreamingPagedMultiHop

ARMS=['lrc_local','lrc_cache','gru','gru_cache','transformer','v4']
class SuffixCache:
 def __init__(self,budget=4096):
  self.budget=budget;self.tables={k:OrderedDict() for k in [2,4,8]};self.history=deque(maxlen=8);self.past=deque(maxlen=128);self.t=0;self.evictions=0
 def observe(self,token,corrupt=False):
  # Write only transitions whose next byte has NOW been observed.
  before=tuple(self.history)
  for k,table in self.tables.items():
   if len(before)>=k:
    key=before[-k:];table[key]=(table.get(key,())+((token,self.t),))[-2:];table.move_to_end(key)
    if len(table)>self.budget:table.popitem(last=False);self.evictions+=1
  self.history.append(token);self.past.append(token);now=tuple(self.history);found=[];length=0
  for k in [8,4,2]:
   if len(now)>=k and now[-k:] in self.tables[k]:
    table=self.tables[k];key=now[-k:];found=list(reversed(table[key]));table.move_to_end(key);length=k;break
  assert all(position<=self.t for _,position in found)
  ids=[value for value,_ in found]
  if corrupt and ids:
   # Deterministic unrelated reads from the observed prefix, never future bytes.
   values=list(self.past);ids=[values[(self.t*17+j*31)%len(values)] for j in range(len(ids))]
  self.t+=1
  return ids+[0]*(2-len(ids)),length
 def allocated_bytes(self):
  seen=set()
  def size(x):
   if id(x) in seen:return 0
   seen.add(id(x));s=sys.getsizeof(x)
   if isinstance(x,dict):s+=sum(size(k)+size(v) for k,v in x.items())
   elif isinstance(x,(tuple,list,deque)):s+=sum(size(y) for y in x)
   return s
  return size(self.tables)+size(self.history)+size(self.past)
def cache_batch(x,corrupt=False):
 ids=[];lengths=[]
 for row in x.tolist():
  c=SuffixCache();pairs=[c.observe(t,corrupt) for t in row];ids.append([z[0] for z in pairs]);lengths.append([z[1] for z in pairs])
 return torch.tensor(ids,device=x.device),torch.tensor(lengths,device=x.device)
class GRU(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(257,24);self.rnn=nn.GRU(24,68,batch_first=True);self.head=nn.Linear(68,257)
 def forward(self,x):return self.head(self.rnn(self.embedding(x))[0])
class Layer(nn.Module):
 def __init__(self):
  super().__init__();d=36;self.n1=nn.RMSNorm(d,eps=1e-6);self.n2=nn.RMSNorm(d,eps=1e-6);self.qkv=nn.Linear(d,3*d,bias=False);self.o=nn.Linear(d,d,bias=False);self.up=nn.Linear(d,64,bias=False);self.gate=nn.Linear(d,64,bias=False);self.down=nn.Linear(64,d,bias=False)
 def rotate(self,z,pos):
  freq=1/(10000**(torch.arange(0,12,2,device=z.device,dtype=z.dtype)/12));a=pos.to(z.dtype)[:,None]*freq[None,:];co,si=a.cos()[None,None],a.sin()[None,None];even,odd=z[...,::2],z[...,1::2]
  return torch.stack((even*co-odd*si,even*si+odd*co),-1).flatten(-2)
 def forward(self,x,cache=None,position=0,collect=False):
  b,n,_=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,n,3,3,12).permute(2,0,3,1,4).unbind(0);pos=torch.arange(position,position+n,device=x.device);q,k=self.rotate(q,pos),self.rotate(k,pos)
  if cache is not None:
   oldk,oldv=cache
   if oldk is not None:k=torch.cat([oldk,k],-2);v=torch.cat([oldv,v],-2)
   k,v=k[:,:,-128:],v[:,:,-128:];attn=F.scaled_dot_product_attention(q,k,v);newcache=(k,v)
  else:
   ix=torch.arange(n,device=x.device);mask=(ix[:,None]>=ix[None,:]) & ((ix[:,None]-ix[None,:])<128);attn=F.scaled_dot_product_attention(q,k,v,attn_mask=mask);newcache=(k[:,:,-128:],v[:,:,-128:]) if collect else None
  x=x+self.o(attn.transpose(1,2).reshape(b,n,36));z=self.n2(x);x=x+self.down(F.silu(self.gate(z))*self.up(z));return x,newcache
class Transformer(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(257,36);self.layers=nn.ModuleList([Layer(),Layer()]);self.norm=nn.RMSNorm(36,eps=1e-6);self.head=nn.Linear(36,257)
 def forward(self,x):
  h=self.embedding(x)
  for layer in self.layers:h,_=layer(h)
  return self.head(self.norm(h))
class Model(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.is_lrc=arm in ('lrc_local','lrc_cache','v4');self.cached=arm.endswith('_cache');self.cache_data=None;self.last_gate=None
  self.base=PagedMultiHopLM(Config(width=24,layers=2,local_kernel=3),page_bits=8) if self.is_lrc else (Transformer() if arm=='transformer' else GRU())
  if arm!='transformer' and arm!='v4':
   self.admission=nn.Linear(4,1);nn.init.zeros_(self.admission.weight);nn.init.constant_(self.admission.bias,-2.)
 def mix(self,logits,ids,lengths,force=None):
  p=logits.softmax(-1);valid=ids!=0;count=valid.sum(-1);prob=p.gather(-1,ids)*valid
  f=torch.stack((lengths/8,(ids[...,0]==ids[...,1]).float()*(count==2),prob.sum(-1)/count.clamp_min(1),torch.log1p(count)/math.log(3)),-1)
  g=self.admission(f).sigmoid().squeeze(-1)*(count>0)
  if force=='disabled':g=g*0
  if force=='fixed':g=torch.full_like(g,torch.sigmoid(torch.tensor(-2.)).item())*(count>0)
  mixed=p*(1-g[...,None]);mixed=mixed.scatter_add(-1,ids,g[...,None]*valid/count.clamp_min(1)[...,None]);self.last_gate=g.detach()
  return mixed.clamp_min(torch.finfo(p.dtype).tiny).log()
 def forward(self,x,force=None,**kw):
  if self.is_lrc:
   if self.arm!='v4':kw['memory']='disabled'
   out=self.base(x,**kw)
  else:out=SimpleNamespace(logits=self.base(x),metrics={})
  if self.cached:
   ids,lengths=self.cache_data if self.cache_data is not None else cache_batch(x)
   out.logits=self.mix(out.logits,ids,lengths,force)
  return out
class Stream:
 def __init__(self,model):
  self.m=model;self.t=0;self.embeds=deque(maxlen=2);self.gru_state=None;self.kv=[(None,None),(None,None)];self.lexical=SuffixCache() if model.cached else None;self.v4=StreamingPagedMultiHop(model.base) if model.arm=='v4' else None
 def local(self,token):
  b=self.m.base;e=b.embedding(token);w=list(self.embeds)
  w=[e.new_zeros(e.shape)]*(2-len(w))+w
  h=b.local_norm(e+F.gelu(F.conv1d(torch.stack(w+[e],-1),b.local.weight,b.local.bias).squeeze(-1)));self.embeds.append(e)
  for layer in b.layers:
   zero=torch.zeros_like(h);routes=layer.route(torch.cat((h,zero),-1)).reshape(1,3,b.config.relations).argmax(-1);ef,es,op=[layer.relation(routes[:,i]) for i in range(3)]
   features=torch.cat((h,zero,zero,ef,es,op,h.new_zeros((1,2))),-1);h=layer.norm(h+layer.compose(features))
  return b.lm_head(h)
 @torch.no_grad()
 def step(self,token,corrupt=False,force=None):
  x=torch.tensor([token]);b=self.m.base
  if self.m.arm=='v4':z=self.v4.step(x)
  elif self.m.is_lrc:z=self.local(x)
  elif self.m.arm=='transformer':
   h=b.embedding(x[:,None])
   for i,layer in enumerate(b.layers):h,self.kv[i]=layer(h,self.kv[i],self.t)
   z=b.head(b.norm(h))[:,0]
  else:
   h,self.gru_state=b.rnn(b.embedding(x[:,None]),self.gru_state);z=b.head(h[:,0])
  if self.lexical:
   ids,k=self.lexical.observe(token,corrupt);z=self.m.mix(z[:,None],torch.tensor([[ids]]),torch.tensor([[k]]),force)[:,0]
  else:z=z.log_softmax(-1)
  self.t+=1;return z
 @torch.no_grad()
 def prefill(self,tokens):
  if self.t or not tokens:raise ValueError('prefill requires fresh stream and nonempty tokens')
  b=self.m.base;x=torch.tensor(tokens)[None]
  if self.m.arm=='v4':z=self.v4.prefill(x)
  elif self.m.is_lrc:
   # Exact finite-receptive-field fast path: only last3 bytes affect the
   # local model's last logits; preserve last2 embeddings for continuation.
   for token in tokens[-3:]:z=self.local(torch.tensor([token]))
  elif self.m.arm=='transformer':
   h=b.embedding(x)
   for i,layer in enumerate(b.layers):h,self.kv[i]=layer(h,collect=True)
   z=b.head(b.norm(h[:,-1]))
  else:
   h,self.gru_state=b.rnn(b.embedding(x));z=b.head(h[:,-1])
  if self.lexical:
   for token in tokens:ids,k=self.lexical.observe(token)
   z=self.m.mix(z[:,None],torch.tensor([[ids]]),torch.tensor([[k]]))[:,0]
  else:z=z.log_softmax(-1)
  self.t=len(tokens);return z
 def state_bytes(self):
  # Snapshot retained state, including Python containers and tensor storage.
  # This is not process RSS or transient peak allocation.
  seen=set();storages=set()
  def size(x):
   if x is None or id(x) in seen:return 0
   seen.add(id(x));n=sys.getsizeof(x)
   if isinstance(x,torch.Tensor):
    st=x.untyped_storage();key=st.data_ptr()
    if key not in storages:storages.add(key);n+=st.nbytes()
   elif isinstance(x,dict):n+=sum(size(k)+size(v) for k,v in x.items())
   elif isinstance(x,(list,tuple,deque)):n+=sum(size(v) for v in x)
   elif hasattr(x,'__dict__'):n+=size(x.__dict__)
   return n
  return size([self.embeds,self.gru_state,self.kv,self.lexical,
               self.v4.cache if self.v4 else None])
