"""End-to-end research candidates. No operation/identity oracle at inference."""
import torch
from torch import nn
from native_csm import RecordRewrite,Config,RecordState,hard_choice

class SeparatedCSM(RecordRewrite):
    """Controlled architectural candidate: explicit payload transports, learned selection.
    Branches are engineered. No claim the policy discovers SET/COPY semantics.
    Input-only encoding removes old-state dependence on one selectable write path.
    """
    def __init__(self,cfg):
        super().__init__(cfg)
        self.payload_encoder=nn.Sequential(nn.Linear(cfg.width,cfg.width),nn.Tanh())
    def step(self,token_ids,state,interventions=None,return_trace=False):
        if return_trace: raise NotImplementedError('trace not admitted')
        raw=self.embedding(token_ids)
        front=torch.tanh(self.frontend(torch.cat((state.history.flatten(1),raw),-1)))
        active=self._read(state.cursor,state.values)
        control=torch.tanh(self.controller(torch.cat((front,active),-1)))
        choose=lambda z:hard_choice(z,self.config.temperature)
        a=choose(self.first_address(control)); va=self._read(a,state.values)
        b0=choose(self.second_address(torch.cat((control,va),-1)))
        linked=self._read(a,state.links); use=choose(self.use_link(control))
        b=use[:,:1]*b0+use[:,1:]*linked;vb=self._read(b,state.values)
        op=choose(self.operator(control)); ov=op@self.operator_embedding
        transformed=self.transform(torch.cat((front,va,vb,ov),-1))
        branches=torch.stack((self.payload_encoder(front),va,vb,transformed),1)
        proposal=(op.unsqueeze(-1)*branches).sum(1)
        dst=choose(self.destination(control));previous=self._read(dst,state.values)
        gate=choose(self.value_edit(control))[:,1:]*torch.sigmoid(self.write_gate(control))
        updated=previous+gate*(proposal-previous)
        values=state.values*(1-dst.unsqueeze(-1))+dst.unsqueeze(-1)*updated.unsqueeze(1)
        edit=choose(self.link_edit(control))[:,1:];mask=dst.unsqueeze(-1)*edit.unsqueeze(-1)
        links=state.links*(1-mask)+mask*b.unsqueeze(1)
        return self.output(updated),RecordState(values,links,dst,torch.stack((state.history[:,1],raw),1),state.position+1)

class LocalCSM(RecordRewrite):
    def step(self,token_ids,state,interventions=None,return_trace=False):
        # Keep the two-byte/token frontend history; discard persistent records/links.
        fresh=self.init_state(token_ids.shape[0],token_ids.device)
        fresh.history=state.history;fresh.position=state.position
        return super().step(token_ids,fresh,interventions,return_trace)

class Transformer(nn.Module):
    def __init__(self,vocab=256,width=128,layers=2,max_context=16384):
        super().__init__();self.emb=nn.Embedding(vocab,width);self.register_buffer("positions",self._positions(max_context,width),persistent=False)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,4,4*width,dropout=0.,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.out=nn.Linear(width,vocab)
    @staticmethod
    def _positions(n,d):
        import math
        p=torch.arange(n).unsqueeze(1);freq=torch.exp(torch.arange(0,d,2)*(-math.log(10000.)/d));z=torch.zeros(n,d);z[:,0::2]=torch.sin(p*freq);z[:,1::2]=torch.cos(p*freq);return z
    def forward(self,x):
        h=self.emb(x)+self.positions[:x.shape[1]];mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        for b in self.blocks:h=b(h,src_mask=mask)
        return self.out(self.norm(h))

def make(arm,vocab=256,width=128,slots=16):
    cfg=Config(vocab,width=width,slots=slots,inner=width*4)
    return {'native':RecordRewrite,'separated':SeparatedCSM,'memory_off':LocalCSM}[arm](cfg) if arm!='transformer' else Transformer(vocab,width=width)
