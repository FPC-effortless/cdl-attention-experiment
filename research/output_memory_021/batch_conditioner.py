"""Ragged prefix streaming for independent frozen native states."""
import torch
from conditioner import flatten_state
class BatchedFrozenSession:
    def __init__(self,model,size):
        self.model=model;self.size=size;self.device=next(model.parameters()).device
        if getattr(getattr(model,'config',None),'mode','rewrite')!='rewrite':raise ValueError('only position-independent rewrite mode supported')
        self.state=model.init_state(size,self.device);self.ingested_bytes=[0]*size
    @torch.no_grad()
    def consume(self,chunks):
        if len(chunks)!=self.size or any(not isinstance(x,bytes) for x in chunks):raise ValueError('byte batch shape')
        for pos in range(max(map(len,chunks),default=0)):
            mask=torch.tensor([pos<len(x) for x in chunks],device=self.device)
            tokens=torch.tensor([x[pos] if pos<len(x) else 0 for x in chunks],device=self.device)
            _,new=self.model.step(tokens,self.state)
            for key in ('values','links','cursor','history'):
                old=getattr(self.state,key);value=getattr(new,key)
                setattr(new,key,torch.where(mask.reshape(self.size,*([1]*(value.ndim-1))),value,old))
            self.state=new
        self.ingested_bytes=[n+len(x) for n,x in zip(self.ingested_bytes,chunks)]
    def feature(self):return flatten_state(self.state)

class SerialFrozenBatch:
    """Slow reference retaining one independent FrozenSession per lane."""
    def __init__(self,model,size):
        from conditioner import FrozenSession
        self.sessions=[FrozenSession(model) for _ in range(size)]
    def consume(self,chunks):
        if len(chunks)!=len(self.sessions):raise ValueError('batch shape')
        for s,c in zip(self.sessions,chunks):s.consume(c)
    def feature(self):return torch.cat([s.feature() for s in self.sessions])
