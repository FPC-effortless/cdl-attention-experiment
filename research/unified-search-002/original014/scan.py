import torch

def sequential(z,a,s=None):
 if s is None:s=torch.zeros_like(z[:,0])
 ys=[]
 for j in range(z.shape[1]):s=a[:,j]*s+(1-a[:,j])*z[:,j];ys.append(s)
 return torch.stack(ys,1),s

def division(z,a,s=None,chunk=16):
 if s is None:s=torch.zeros_like(z[:,0])
 yy=[]
 for x,g in zip(z.split(chunk,1),a.split(chunk,1)):
  c=g.cumprod(1);y=c*(s[:,None]+(((1-g)*x)/c).cumsum(1));s=y[:,-1];yy.append(y)
 return torch.cat(yy,1),s

def associative(z,a,s=None,chunk=16,accum=None):
 if accum is not None:z=z.to(accum);a=a.to(accum)
 if s is None:s=torch.zeros_like(z[:,0])
 else:s=s.to(z.dtype)
 yy=[]
 for x,g in zip(z.split(chunk,1),a.split(chunk,1)):
  A=g;B=(1-g)*x;offset=1
  while offset<x.shape[1]:
   A,B=torch.cat((A[:,:offset],A[:,offset:]*A[:,:-offset]),1),torch.cat((B[:,:offset],B[:,offset:]+A[:,offset:]*B[:,:-offset]),1);offset*=2
  y=A*s[:,None]+B;s=y[:,-1];yy.append(y)
 return torch.cat(yy,1),s
