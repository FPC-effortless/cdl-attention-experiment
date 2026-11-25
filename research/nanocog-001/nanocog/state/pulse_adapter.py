"""Diagnostic integration with PULSE006 native sentence, broker and proposal heads."""
import torch
from torch import nn
from torch.nn import functional as F

from .records import HOLD, SET, TypedRecordBoard, WritePacket


class Pulse006TypedAdapter(nn.Module):
    """Counterfactual replacement, not a drop-in pretrained semantic reader.

    Reuses actual PULSE006 encoder, observation board, one optional second
    subscriber, merge_event, commit/address/effect and lookup. Native trained
    content-derived second lookup and answer classifier are replaced by discrete
    typed readback of the fixed 005 taught coordinate meanings. A wrong typed
    proposal may be rejected, never corrected using teacher metadata.
    """
    _required=("enc pub writer msg sub ask broker merge_event commit addr effect lookup".split())

    def __init__(self,pulse):
        super().__init__()
        if any(not hasattr(pulse,k) for k in self._required) or pulse.slots!=8 or pulse.channels!=4:
            raise ValueError("expected upstream PULSE006 architecture")
        self.pulse=pulse
        self.board=TypedRecordBoard()

    def forward(self,x,lengths,*,return_trace=False):
        if x.ndim!=3 or lengths.dtype!=torch.long or lengths.shape!=(x.size(0),) or lengths.device!=x.device:
            raise ValueError("expected PULSE token tensor and int64 lengths")
        if ((lengths<1)|(lengths>x.size(1))).any():raise ValueError("length outside sentence tensor")
        m=self.pulse
        z=m.enc(x); b,_,d=z.shape
        obs=z.new_zeros((b,m.channels,d))
        state=self.board.empty(b,z.device)
        traces=[]
        for j in range(x.shape[1]-1):
            e=z[:,j]; live=j<lengths-1
            pub=(m.pub(e)[:,0]>=0)&live
            writer=m.writer(e).argmax(-1)
            mask=(F.one_hot(writer,m.channels).bool()&pub[:,None])[:,:,None]
            obs=torch.where(mask,m.msg(e)[:,None],obs)
            sub=m.sub(e); first=sub.argmax(-1)
            obs1=obs[torch.arange(b,device=z.device),first]
            ask=(m.ask(obs1)[:,0]>=0)&live
            second=sub.masked_fill(F.one_hot(first,m.channels).bool(),-1e4).argmax(-1)
            obs2=obs[torch.arange(b,device=z.device),second].masked_fill(~ask[:,None],0)
            event=m.merge_event(e,m.broker(torch.cat((obs1,obs2),dim=-1)))
            # Native taught coordinate convention is diagnostic data interface only.
            # Keep unmasked argmax so invalid person/box values are detected, not repaired.
            a=m.addr(event).argmax(-1)
            v=m.effect(event)[:,:self.board.values].argmax(-1)
            commit=pub if m.tied else (m.commit(event)[:,0]>=0)&live
            op=torch.where(commit,torch.full_like(a,SET),torch.full_like(a,HOLD))
            state,rec=self.board.apply(state,WritePacket(a,op,v),live)
            if return_trace:traces.append((a,v,commit,rec.legal,rec.changed))
        # The final question is not an event and cannot modify either board.
        query=z[torch.arange(b,device=z.device),lengths-1]
        address=m.lookup(query).argmax(-1)
        result=self.board.read(state,address)
        return (result,state,traces) if return_trace else result
