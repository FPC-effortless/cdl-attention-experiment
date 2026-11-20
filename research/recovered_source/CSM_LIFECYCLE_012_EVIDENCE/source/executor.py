"""Exact batched CPU typed store; integer payloads only, no learned policy claim."""
import torch
import torch.nn.functional as F

def init(batch, n=4):
 if type(batch) is not int or batch<1 or type(n) is not int or n<1: raise ValueError('positive batch/capacity')
 return torch.zeros(batch,n,dtype=torch.long),torch.zeros(batch,n,dtype=torch.long)

def validate(state):
 k,d=state
 if k.dtype!=torch.long or d.dtype!=torch.long or k.ndim!=2 or d.shape!=k.shape or min(k.shape)<1 or k.device!=d.device:raise ValueError('state shape/type/device')
 if not bool(((k>=0)&(k<=2)).all()):raise ValueError('kind')
 if not bool(torch.where(k==1,(d>=1)&(d<=4),torch.where(k==2,(d>=0)&(d<k.shape[1]),d==0)).all()):raise ValueError('data')
 return k,d

def _read(state,ids):
 k,d=state;b,n=k.shape;rows=torch.arange(b,device=k.device);idx=ids.clone();live=torch.ones(b,dtype=torch.bool,device=k.device);out=torch.zeros(b,dtype=torch.long,device=k.device)
 for _ in range(n):
  kinds=k[rows,idx];data=d[rows,idx];hit=live&(kinds==1);out=torch.where(hit,data,out);live=live&(kinds==2);idx=torch.where(live,data,torch.zeros_like(idx))
 return out

def step(state,op,dst,src,value):
 k,d=validate(state);b,n=k.shape
 for a in (op,dst,src,value):
  if a.dtype!=torch.long or a.shape!=(b,) or a.device!=k.device:raise ValueError('action shape/type/device')
 if not bool(((op>=0)&(op<=5)).all()) or not bool(((dst>=0)&(dst<n)).all()):raise ValueError('operation/destination')
 needsrc=(op==2)|(op==3)
 if not bool((~needsrc|((src>=0)&(src<n))).all()):raise ValueError('source')
 if not bool(((op!=1)|((value>=1)&(value<=4))).all()):raise ValueError('value')
 safe_src=torch.where(needsrc,src,torch.zeros_like(src));effective=_read(state,safe_src);answer=torch.where(op==5,_read(state,dst),torch.full_like(op,-1));nk=k.clone();nd=d.clone();rows=torch.arange(b,device=k.device)
 replace=(op>=1)&(op<=4);newkind=torch.where(op==3,2,torch.where((op==1)|((op==2)&(effective>0)),1,0));newdata=torch.where(op==1,value,torch.where(op==2,effective,torch.where(op==3,safe_src,0)))
 nk[rows,dst]=torch.where(replace,newkind,k[rows,dst]);nd[rows,dst]=torch.where(replace,newdata,d[rows,dst]);return (nk,nd),answer

def features(state):
 k,d=validate(state)
 if k.shape[1]!=4:raise ValueError('features require four identities')
 kinds=F.one_hot(k,3);values=F.one_hot((d-1).clamp(0,3),4)*(k==1).unsqueeze(-1);refs=F.one_hot(d.clamp(0,3),4)*(k==2).unsqueeze(-1)
 return torch.cat((kinds,values,refs),-1).flatten(1).float()

def read(state,ids):
 k,d=validate(state)
 if ids.dtype!=torch.long or ids.shape!=(k.shape[0],) or ids.device!=k.device or not bool(((ids>=0)&(ids<k.shape[1])).all()):raise ValueError('read indices')
 return _read(state,ids)
