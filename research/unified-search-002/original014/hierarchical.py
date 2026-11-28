import torch

def _prefix(A,B,dim):
 offset=1;n=A.shape[dim]
 def head(x):return x.narrow(dim,0,offset)
 def right(x):return x.narrow(dim,offset,n-offset)
 def left(x):return x.narrow(dim,0,n-offset)
 while offset<n:
  A,B=torch.cat((head(A),right(A)*left(A)),dim),torch.cat((head(B),right(B)+right(A)*left(B)),dim)
  offset*=2
 return A,B

def hierarchical(z,a,s=None,chunk=16,accum=None):
 if accum is not None:z=z.to(accum);a=a.to(accum)
 if s is None:s=torch.zeros_like(z[:,0])
 else:s=s.to(z.dtype)
 n=z.shape[1];p=(-n)%chunk
 if p:
  z=torch.cat([z,torch.zeros_like(z[:,:1]).expand(-1,p,*z.shape[2:])],1)
  a=torch.cat([a,torch.ones_like(a[:,:1]).expand(-1,p,*a.shape[2:])],1)
 B=((1-a)*z).unflatten(1,(-1,chunk));A=a.unflatten(1,(-1,chunk));A,B=_prefix(A,B,2);AA,BB=_prefix(A[:,:,-1],B[:,:,-1],1)
 end=AA*s[:,None]+BB;start=torch.cat((s[:,None],end[:,:-1]),1);y=(A*start[:,:,None]+B).flatten(1,2)[:,:n]
 return y,y[:,-1]
