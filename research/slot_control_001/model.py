"""Conventional learned discrete slot store. No semantic metadata in forward."""
import torch
from torch import nn
from torch.nn import functional as F

def choose(logits):
    probabilities=logits.softmax(-1)
    hard=F.one_hot(probabilities.argmax(-1),probabilities.shape[-1]).to(logits.dtype)
    return hard+(probabilities-probabilities.detach())

class SlotStore(nn.Module):
    def __init__(self,vocab_size=19,width=32,slots=8,inner=208):
        super().__init__(); self.width=width;self.slots=slots
        self.embedding=nn.Embedding(vocab_size,width)
        self.frontend=nn.Linear(3*width,width)
        self.destination=nn.Linear(width,slots)
        self.value_edit=nn.Linear(width,2)
        self.write_gate=nn.Linear(width,1)
        self.proposal=nn.Sequential(nn.Linear(width,inner),nn.Tanh(),nn.Linear(inner,width),nn.Tanh())
        self.output=nn.Linear(width,vocab_size)
        nn.init.constant_(self.write_gate.bias,-.5)
        with torch.no_grad(): self.value_edit.bias.copy_(torch.tensor([0.,.3]))

    def init_state(self,batch,device):
        return (torch.zeros(batch,self.slots,self.width,device=device,dtype=self.embedding.weight.dtype),
                torch.zeros(batch,2,self.width,device=device,dtype=self.embedding.weight.dtype))

    def step(self,ids,state,collect=False):
        values,history=state;raw=self.embedding(ids)
        front=torch.tanh(self.frontend(torch.cat((history.flatten(1),raw),-1)))
        dest=self.destination(front);edit=self.value_edit(front);gain=self.write_gate(front)
        address=choose(dest);admission=choose(edit)[:,1:]
        previous=torch.bmm(address[:,None],values).squeeze(1)
        updated=previous+admission*torch.sigmoid(gain)*(self.proposal(front)-previous)
        values=values*(1-address[:,:,None])+address[:,:,None]*updated[:,None]
        logits=self.output(updated)
        state=(values,torch.stack((history[:,1],raw),1))
        if collect:return logits,state,{'destination':dest,'value_edit':edit,'write_gate':gain}
        return logits,state

    def sequence(self,ids,collect=False,all_outputs=False):
        state=self.init_state(ids.shape[0],ids.device);captures=[];outputs=[]
        for token in ids.unbind(1):
            if collect: logits,state,cap=self.step(token,state,True);captures.append(cap)
            else: logits,state=self.step(token,state)
            if all_outputs:outputs.append(logits)
        cap={k:torch.stack([c[k] for c in captures],1) for k in captures[0]} if captures else {}
        return torch.stack(outputs,1) if all_outputs else logits,cap
