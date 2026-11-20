import torch
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
  ls.append(F.cross_entropy(p[mask],y[:,:,i][mask],weight=weights if i==0 else None) if bool(mask.any()) else torch.where(torch.isfinite(p),p,torch.zeros_like(p)).sum()*0)
 return sum(ls)
