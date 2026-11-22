"""Two causal token mixers. The caller owns residual/norm and branch composition."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalState:
    # Cached inputs of up to window-1 positions, oldest first.
    history: torch.Tensor

    def detach(self):
        return LocalState(self.history.detach())

    def index_select(self, ids):
        return LocalState(self.history.index_select(0,ids))


@dataclass(frozen=True)
class SummaryState:
    # [B,2,C]: sum(weight * value), sum(weight), for complete and pending packets.
    complete: torch.Tensor
    pending: torch.Tensor
    phase: int = 0

    def detach(self):
        return SummaryState(self.complete.detach(),self.pending.detach(),self.phase)

    def index_select(self, ids):
        return SummaryState(self.complete.index_select(0,ids),self.pending.index_select(0,ids),self.phase)


def _size(width,window,name):
    if type(width) is not int or width<1 or type(window) is not int or window<1:
        raise ValueError("positive integer width/"+name+" required")


def _shape(x,width):
    if x.ndim!=3 or x.shape[0]<1 or x.shape[-1]!=width or not x.is_floating_point():
        raise ValueError("expected floating [batch,time,width] with nonempty batch")
    return x.shape[:2]


class LocalMix(nn.Module):
    """Token-conditioned offset gates over factorized depthwise shifted taps.

    The gate sees only the current input and emits weights over relative offset,
    not scores against any other token. Each offset has per-channel taps, and
    the aggregate passes through a single shared pointwise projection. This is
    not Q/K/V attention. The tap band includes the current position.
    """
    def __init__(self,width,window=4):
        super().__init__()
        _size(width,window,"window")
        self.width=width
        self.window=window
        self.offset_gate=nn.Linear(width,window)
        self.depthwise=nn.Parameter(torch.ones(width,window))
        self.pointwise=nn.Linear(width,width,bias=False)

    def forward(self,x,state=None):
        b,t=_shape(x,self.width)
        if state is None:
            state=LocalState(x[:,:0].clone())
        if (not isinstance(state,LocalState) or state.history.ndim!=3 or
                state.history.shape!=(b,state.history.size(1),self.width) or
                state.history.size(1)>=self.window or state.history.dtype!=x.dtype or
                state.history.device!=x.device):
            raise ValueError("invalid LocalMix state")
        if not t:return x[:,:0],state
        h=state.history.size(1)
        samples=torch.cat((state.history,x),dim=1)
        missing=self.window-1-h
        taps=F.pad(samples,(0,0,missing,0)).unfold(1,self.window,1)
        score=self.offset_gate(x)
        ident=torch.arange(t,device=x.device)[:,None]+torch.arange(self.window,device=x.device)[None,:]
        gate=score.masked_fill(ident<missing,-torch.inf).softmax(-1)
        mixture=(taps*self.depthwise.to(taps)*gate.unsqueeze(-2).to(taps)).sum(-1)
        delta=self.pointwise(mixture)
        last=self.window-1
        # Clone after slicing to avoid retaining the full concatenation's storage.
        keep=samples[:, -last:] if last else samples[:,:0]
        return delta,LocalState(keep.clone())


class SummaryMix(nn.Module):
    """Token → block moments → exclusive earlier-block prefix → query gate.

    Per-channel positive token weights are learned from the current input; each
    contiguous block contributes a two-vector packet (numerator, denominator).
    Their merge is addition. A token can only read the packet prefix before the
    start of its own block. The current block, including earlier positions in it,
    never contributes to the current read. This module returns summary-only delta.

    The stream state is fixed size, independent of sequence length, and contains
    no raw previous tokens. All rows share one phase (unpadded input).
    """
    def __init__(self,width,block=8):
        super().__init__()
        _size(width,block,"block")
        self.width=width
        self.block=block
        self.weight=nn.Linear(width,width)
        self.value=nn.Linear(width,width,bias=False)
        self.query=nn.Linear(width,width)
        self.out=nn.Linear(width,width,bias=False)

    def forward(self,x,state=None):
        b,t=_shape(x,self.width)
        accum=torch.promote_types(torch.float32,x.dtype)
        if state is None:
            z=torch.zeros((b,2,self.width),device=x.device,dtype=accum)
            state=SummaryState(z,z.clone())
        if (not isinstance(state,SummaryState) or type(state.phase) is not int or
                not 0<=state.phase<self.block or
                any(u.shape!=(b,2,self.width) or u.dtype!=accum or u.device!=x.device
                    for u in (state.complete,state.pending))):
            raise ValueError("invalid SummaryMix state")
        if not t:return x[:,:0],state
        weight=self.weight(x).to(accum).sigmoid()
        weighted=self.value(x).to(accum)*weight
        moment=torch.stack((weighted,weight),dim=2)
        # Align the beginning of this chunk with its persistent global block.
        # Missing leading events are represented by one already accumulated packet,
        # not fabricated tokens. Trailing zeros are excluded by commit count.
        total_len=state.phase+t
        nfull,phase=divmod(total_len,self.block)
        last_pad=(-total_len)%self.block
        packed=F.pad(moment,(0,0,0,0,state.phase,last_pad))
        blocks=packed.reshape(b,-1,self.block,2,self.width).sum(2)
        blocks=torch.cat((blocks[:,:1]+state.pending[:,None],blocks[:,1:]),dim=1)
        # Index j<Tblocks receives exactly the sums of blocks strictly before j.
        prefix=torch.cat((state.complete[:,None],state.complete[:,None]+blocks.cumsum(dim=1)),dim=1)
        ix=(state.phase+torch.arange(t,device=x.device))//self.block
        eligible=prefix.index_select(1,ix)
        num,den=eligible.unbind(2)
        mean=torch.where(den>0,num/den.clamp_min(torch.finfo(accum).tiny),0)
        delta=self.out(mean.to(x.dtype)*self.query(x).sigmoid())
        # Clone the selected packets, releasing large transient prefill storage.
        complete=prefix[:,nfull].clone()
        pending=blocks[:,nfull].clone() if phase else torch.zeros_like(complete)
        return delta,SummaryState(complete,pending,phase)
