"""Optional final local value residual, matched content and literal-lag arms."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('tap_content','literal_lag')
class _BitwiseAdd(torch.autograd.Function):
    """Use addition's derivative, preserving exact zero and masked native bits."""
    @staticmethod
    def forward(ctx,native,delta):
        return torch.where(delta==0,native,native+delta)
    @staticmethod
    def backward(ctx,grad):
        return grad,grad

def exact_residual_add(native,delta):
    if native.shape!=delta.shape or native.dtype!=delta.dtype or native.device!=delta.device:
        raise ValueError('native mixture geometry differs')
    return _BitwiseAdd.apply(native,delta)

class ValuePathResidual(nn.Module):
    """Pure final-row function on raw literal query and native local tensors.

    q [B,C], taps [B,4,C] (pre-depthwise), gate [B,4] native probabilities.
    Requires no answer labels, roles, routing metadata, or sibling examples.
    """
    def __init__(self,width=32,hidden=8,*,arm='tap_content',bound=.25):
        super().__init__()
        if width<4 or hidden<1 or arm not in ARMS or bound!=.25:
            raise ValueError('unregistered residual geometry or bound')
        self.width=width;self.arm=arm;self.bound=bound
        self.first=nn.Linear(2*width+4,hidden,bias=False)
        self.last=nn.Linear(hidden,width,bias=False)
        nn.init.zeros_(self.last.weight)
        j=torch.arange(width);alt=1-2*(j%2)
        self.register_buffer('lag_onehot',torch.eye(4),persistent=False)
        self.register_buffer('literal_permutation',torch.stack((j,j,(j-1)%width,(j-3)%width)),persistent=False)
        self.register_buffer('literal_sign',torch.stack((torch.ones_like(j),-torch.ones_like(j),alt,-alt)),persistent=False)
    def features(self,q,taps):
        if q.ndim!=2 or q.shape[-1]!=self.width or q.dtype!=torch.float32:
            raise ValueError('literal raw query must be [B,C]')
        b,c=q.shape
        if taps.shape!=(b,4,c) or taps.device!=q.device or taps.dtype!=q.dtype:
            raise ValueError('native tap geometry')
        if not (bool(torch.isfinite(q).all()) and bool(torch.isfinite(taps).all())):
            raise ValueError('nonfinite native features')
        q=F.normalize(q.detach(),dim=-1,eps=1e-6)
        # Both arms have these exact shapes. I/-I across lags avoids exact
        # universal collapse of Wq and Wtap into a single query projection.
        if self.arm=='tap_content':
            src=taps.detach()
        else:
            src=q[:,self.literal_permutation]*self.literal_sign
        u=F.normalize(src,dim=-1,eps=1e-6)
        return torch.cat((q.unsqueeze(1).expand(b,4,c),u,self.lag_onehot.to(q).expand(b,4,4)),dim=-1)
    def forward(self,q,taps,gate):
        if gate.shape!=(q.size(0),4) or gate.dtype!=q.dtype or gate.device!=q.device:
            raise ValueError('native gate geometry')
        w=gate.detach()
        if not (bool(torch.isfinite(w).all()) and bool((w>=0).all()) and
                bool((w.double().sum(-1)-1.).abs().le(2e-7).all())):
            raise ValueError('expected native normalized probabilities')
        corrections=self.last(F.silu(self.first(self.features(q,taps)))).tanh()*self.bound
        # Apply unchanged native weights; do not add or infer a gate.
        out=(corrections*w.unsqueeze(-1)).sum(1)
        if not bool(torch.isfinite(out).all()):raise ValueError('nonfinite correction')
        return out
    def apply_to(self,native_last,q,taps,gate,*,enabled=False):
        if not enabled:return native_last
        return exact_residual_add(native_last,self(q,taps,gate))
