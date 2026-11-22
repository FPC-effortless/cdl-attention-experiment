"""Matched optional ROLE-REPAIR-004 logit residual; no dependency on nanoCOG."""
import torch
from torch import nn
from torch.nn import functional as F

class _ExactAdd(torch.autograd.Function):
    """Preserve original bits at exact-zero updates, with addition's derivative."""
    @staticmethod
    def forward(ctx,native,delta):
        return torch.where(delta == 0,native,native+delta)
    @staticmethod
    def backward(ctx,grad):
        return grad,grad

class TapGateResidual(nn.Module):
    """Explicit raw query and causal, already-provided taps; no state or targets.

    q: B,T,C, raw literal token embedding from the frozen input embedding.
    taps: B,T,4,C, existing layer inputs as routed by LocalMix, oldest first.
    eligible: T,4 or B,T,4, observed causal geometry only.
    active: optional T or B,T Boolean positions fixed by the caller's contract.
    """
    K=4
    ARMS=('tap_compare','literal_control')
    def __init__(self,width=32,hidden=8,*,arm='tap_compare',max_delta=4.0):
        super().__init__()
        if width<4 or hidden<1 or max_delta<=0 or arm not in self.ARMS:
            raise ValueError('invalid residual design')
        self.width=width;self.hidden=hidden;self.arm=arm;self.max_delta=float(max_delta)
        self.first=nn.Linear(2*width+self.K,hidden,bias=False)
        self.last=nn.Linear(hidden,1,bias=False)
        nn.init.zeros_(self.last.weight)
        self.register_buffer('lag_onehot',torch.eye(self.K),persistent=False)
        j=torch.arange(width)
        alt=1-2*(j%2)
        # In particular I / -I prevent the universal two-projection
        # collapse that occurs if every lag uses the same fixed transform.
        self.register_buffer('literal_permutation',torch.stack((j,j,(j-1)%width,(j-3)%width)),persistent=False)
        self.register_buffer('literal_sign',torch.stack((torch.ones_like(j),-torch.ones_like(j),alt,-alt)),persistent=False)
    def features(self,q,taps):
        b,t,c=q.shape
        q=F.normalize(q.detach(),dim=-1,eps=1e-6)
        if self.arm=='tap_compare':
            source=taps.detach()
        else:
            # Every lag has its own fixed invertible signed coordinate map.
            # No context, score, labels or extra parameters reach this control.
            source=q[...,self.literal_permutation]*self.literal_sign
        k=F.normalize(source,dim=-1,eps=1e-6)
        lag=self.lag_onehot.to(q).expand(b,t,self.K,self.K)
        return torch.cat((q.unsqueeze(-2).expand(b,t,self.K,c),k,lag),dim=-1)
    def forward(self,q,taps,eligible,*,active=None):
        if q.ndim!=3 or q.shape[-1]!=self.width or not q.is_floating_point():
            raise ValueError('raw literal q must be floating B,T,C')
        b,t,c=q.shape
        if taps.shape!=(b,t,self.K,c) or taps.dtype!=q.dtype or taps.device!=q.device:
            raise ValueError('taps must be same-typed/device B,T,4,C')
        if eligible.shape not in ((t,self.K),(b,t,self.K)) or eligible.dtype!=torch.bool or eligible.device!=q.device:
            raise ValueError('invalid causal eligibility')
        valid=eligible.expand(b,t,self.K)
        if active is not None:
            if active.shape not in ((t,),(b,t)) or active.dtype!=torch.bool or active.device!=q.device:
                raise ValueError('invalid fixed active mask')
            valid=valid & active.expand(b,t).unsqueeze(-1)
        # Mask before normalization as well as after scoring. Invalid
        # padded/future contents then cannot affect values or gradients.
        feat=self.features(q,taps.masked_fill(~valid.unsqueeze(-1),0.))
        score=self.last(F.silu(self.first(feat))).squeeze(-1).tanh().masked_fill(~valid,0.)
        count=valid.sum(dim=-1,keepdim=True).clamp_min(1)
        centered=(score-score.sum(dim=-1,keepdim=True)/count).masked_fill(~valid,0.)
        return centered*(self.max_delta/2)
    def apply_to(self,native,q,taps,eligible,*,active=None,enabled=True):
        if not enabled:
            return native
        delta=self(q,taps,eligible,active=active)
        if native.shape!=delta.shape or native.dtype!=delta.dtype or native.device!=delta.device:
            raise ValueError('residual must exactly match native score geometry')
        return _ExactAdd.apply(native,delta)
