"""Native CSM plus a parameter-matched conventional pointer slot control."""
import torch
from torch import nn
from torch.nn import functional as F
from native_csm import Config, RecordRewrite, RecordState, hard_choice
from data import HEADS, TOKENS


class Instrumented:
    def instrument(self):
        self.capture=None;self.supplied=None;self.position=0
        for name in (*HEADS,'write_gate'):
            getattr(self,name).register_forward_hook(self.hook(name))

    def hook(self,name):
        def observe(module,args,output):
            if self.capture is not None:self.capture[name].append(output)
            if self.supplied is None:return None
            target=self.supplied[name][:,self.position]
            valid=target>=0
            if name=='write_gate':replacement=torch.full_like(output,32)
            else:replacement=F.one_hot(target.clamp_min(0),output.shape[-1]).to(output.dtype)*64-32
            return torch.where(valid[:,None],replacement,output)
        return observe

    def sequence(self,ids,collect=False,mutation=None,supplied=None):
        self.capture={k:[] for k in (*HEADS,'write_gate')} if collect else None
        self.supplied=supplied
        state=self.init_state(ids.shape[0],ids.device)
        try:
            for p,token in enumerate(ids.unbind(1)):
                self.position=p
                if p==ids.shape[1]-3:
                    if mutation=='erase_values':state.values=torch.zeros_like(state.values)
                    elif mutation=='identity_links':state.links=torch.eye(self.config.slots,device=ids.device)[None].expand(ids.shape[0],-1,-1)
                    elif mutation is not None:raise ValueError('unknown intervention')
                    before_values=state.values[:,:4].detach().clone()
                    before_links=state.links[:,:4].detach().clone()
                logits,state=self.step(token,state)
            self.last_preservation=((state.values[:,:4].detach()==before_values).flatten(1).all(1)
                & (state.links[:,:4].detach()==before_links).flatten(1).all(1))
            cap={k:torch.stack(v,1) for k,v in self.capture.items()} if collect else {}
            return logits,cap
        finally:
            self.capture=None;self.supplied=None


class CSM(Instrumented,RecordRewrite):
    def __init__(self):
        super().__init__(Config(vocab_size=len(TOKENS)))
        self.instrument()


class PointerSlot(Instrumented,nn.Module):
    def __init__(self):
        super().__init__();d=32;s=8;v=len(TOKENS)
        self.config=Config(vocab_size=v)
        self.embedding=nn.Embedding(v,d)
        self.frontend=nn.Linear(3*d,d)
        self.first_address=nn.Linear(d,s)
        self.second_address=nn.Linear(2*d,s)
        self.destination=nn.Linear(d,s)
        self.use_link=nn.Linear(d,2)
        self.link_edit=nn.Linear(d,2)
        self.value_edit=nn.Linear(d,2)
        self.write_gate=nn.Linear(d,1)
        self.transform=nn.Sequential(nn.Linear(3*d,98),nn.Tanh(),nn.Linear(98,d),nn.Tanh())
        self.output=nn.Linear(d,v)
        nn.init.constant_(self.write_gate.bias,-.5)
        with torch.no_grad():self.value_edit.bias.copy_(torch.tensor([0.,.3]))
        self.instrument()

    init_state=RecordRewrite.init_state
    _read=staticmethod(RecordRewrite._read)

    def step(self,ids,state):
        raw=self.embedding(ids)
        front=torch.tanh(self.frontend(torch.cat((state.history.flatten(1),raw),-1)))
        first=hard_choice(self.first_address(front))
        va=self._read(first,state.values)
        direct=hard_choice(self.second_address(torch.cat((front,va),-1)))
        linked=self._read(first,state.links)
        use=hard_choice(self.use_link(front))
        second=use[:,:1]*direct+use[:,1:]*linked
        vb=self._read(second,state.values)
        proposal=self.transform(torch.cat((front,va,vb),-1))
        destination=hard_choice(self.destination(front))
        prior=self._read(destination,state.values)
        edit=hard_choice(self.value_edit(front))[:,1:]
        gain=edit*torch.sigmoid(self.write_gate(front))
        updated=prior+gain*(proposal-prior)
        values=state.values*(1-destination[:,:,None])+destination[:,:,None]*updated[:,None]
        link_edit=hard_choice(self.link_edit(front))[:,1:]
        mask=destination[:,:,None]*link_edit[:,:,None]
        links=state.links*(1-mask)+mask*second[:,None]
        state=RecordState(values,links,destination,torch.stack((state.history[:,1],raw),1),state.position+1)
        return self.output(updated),state


def build(arm):
    if arm=='csm':return CSM()
    if arm=='pointer_slot':return PointerSlot()
    raise ValueError('unknown arm')


def tensors(rows):
    return {name:torch.tensor([r[name] for r in rows],dtype=torch.long)
            for name in ('ids','answer',*HEADS,'write_gate')}


def objective(logits,cap,batch):
    parts={'answer':F.cross_entropy(logits,batch['answer'])}
    for name in HEADS:
        parts[name]=F.cross_entropy(cap[name].flatten(0,1),batch[name].flatten(),ignore_index=-100)
    mask=batch['write_gate']>=0
    parts['write_gate']=F.binary_cross_entropy_with_logits(cap['write_gate'].squeeze(-1)[mask],
                                                          batch['write_gate'][mask].float())
    loss=sum(value*(.25 if name=='write_gate' else 1) for name,value in parts.items())
    return loss,{name:float(value.detach()) for name,value in parts.items()}
