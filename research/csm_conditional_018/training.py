import torch
from torch.nn import functional as F
def targets(cases):
 y=torch.full((len(cases),len(cases[0]['text']),5),-100,dtype=torch.long);y[:,:,0]=0
 for i,c in enumerate(cases):
  for e in c['events']:
   t=e['pos'];k=e['op'];y[i,t,0]=k;y[i,t,1]=e['dst']
   if k in (2,3,6):y[i,t,2]=e['src']
   if k in (1,6):y[i,t,3]=e['value']-1
   if k==6:y[i,t,4]=e['expected']-1
 return y
def loss(logits,y,weights,gates):
 pieces=logits.split([7,4,4,4,4,2],-1);total=logits[torch.isfinite(logits)].sum()*0
 for i,p in enumerate(pieces):
  target=gates if i==5 else y[:,:,i];mask=target!=-100
  if mask.any():total=total+F.cross_entropy(p[mask],target[mask],weight=weights if i==0 else None)
 return total
