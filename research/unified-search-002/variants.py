"""Structural layer-wise HH/U search. No full-history attention.
N=no side, S=slow state, R=reflection state, B=both. Physical removal,
not output masking. All retained parent computations are unchanged.
"""
import copy, math, torch
from torch import nn
from torch.nn import functional as F
from model import NativeLM, Side, D, scan, hard, count
MASKS=('N','S','R','B')
ARMS=tuple(a+b for a in MASKS for b in MASKS)
WIDTH={'N':1581,'S':1568,'R':1574,'B':1560}
class PartialSide(nn.Module):
    def __init__(self,source,mode):
        super().__init__();assert mode in ('S','R');self.mode=mode
        self.norm=copy.deepcopy(source.norm)
        self.proj=nn.Linear(8,D,bias=False)
        with torch.no_grad():self.proj.weight.copy_(source.proj.weight[:, :8] if mode=='S' else source.proj.weight[:, 8:])
        if mode=='S':
            self.inp=copy.deepcopy(source.inp);self.register_buffer('retention',source.retention.detach().clone())
        else:
            self.action=copy.deepcopy(source.action);self.axes=nn.Parameter(source.axes.detach().clone());self.initial=nn.Parameter(source.initial.detach().clone())
    def forward(self,h,state=None):
        u=self.norm(h)
        if self.mode=='S':
            z,a=self.inp(u).chunk(2,-1);s,ss=scan(z.tanh(),(a+self.retention).sigmoid(),None if state is None else state[0]);return self.proj(s),(ss,None)
        k=F.normalize(self.axes[0],dim=-1);w=F.normalize(self.axes[1]-(self.axes[1]@k)*k,dim=-1);v=.5*k+math.sqrt(.75)*w;axes=torch.stack([k,v]);p=hard(self.action(u).softmax(-1));eye=torch.eye(8,device=h.device,dtype=h.dtype);Hs=eye[None]-2*axes[:,:,None]*axes[:,None,:];A=torch.einsum('bti,ijk->btjk',p,torch.cat([eye[None],Hs],0));s=self.initial[None].expand(len(h),-1) if state is None else state[1];out=[]
        for block in A.split(64,1):
            j=1
            while j<block.size(1):block=torch.cat([block[:,:j],block[:,j:]@block[:,:-j]],1);j*=2
            y=(block@s[:,None,:,None]).squeeze(-1);s=y[:,-1];out.append(y)
        return self.proj(torch.cat(out,1)),(None,s)
class ModularCOG(NativeLM):
    def __init__(self,pattern,ff,side_sources):
        assert pattern in ARMS;super().__init__('HH',ff=ff);self.pattern=pattern
        self.with_side=pattern!='NN';self.sides=nn.ModuleList([None if c=='N' else copy.deepcopy(side_sources[i]) if c=='B' else PartialSide(side_sources[i],c) for i,c in enumerate(pattern)])
    def forward(self,x,state=None,carry=False,features=False):
        h,last,ax=self.local(x,self.core.emb,None if state is None else state[1]);ss=[];sd=[]
        for i,l in enumerate(self.core.blocks):
            st=None if state is None else state[0][i];extra=None;st_side=None
            if self.sides[i] is not None:extra,st_side=self.sides[i](h,None if state is None else state[2][i])
            h,s=l(h,ax,self.payload_map[i],st,extra);ss.append(s);sd.append(st_side)
        h=self.core.norm(h);z=F.linear(h,self.core.emb.weight);s=(ss,last,sd) if self.with_side else (ss,last)
        if features:return z,s,h
        return (z,s) if carry else z

def copy_backbone(dst,src):
    # Shared weights copied exactly, smaller FF layers receive leading neurons.
    sd=dst.state_dict();ss=src.state_dict()
    for name,t in sd.items():
        if name.startswith('sides.'):continue
        a=ss[name];assert a.ndim==t.ndim and all(x>=y for x,y in zip(a.shape,t.shape)),name
        sd[name]=a[tuple(slice(0,s) for s in t.shape)].clone() if a.ndim else a.clone()
    dst.load_state_dict(sd,strict=True)

def fresh(pattern,seed):
    # Exact common HH initialization; side initialization uses independent paired seed.
    torch.manual_seed(seed);hh=NativeLM('HH')
    if pattern=='NN':return hh
    torch.manual_seed(seed+271828);sources=[Side(),Side()]
    m=ModularCOG(pattern,tuple(WIDTH[c] for c in pattern),sources);copy_backbone(m,hh);return m

def from_unified(u,pattern):
    # Exact U-width structural counterpart to frozen output ablation.
    m=ModularCOG(pattern,u.ff,u.sides);copy_backbone(m,u);return m

def tensor_bytes(x):
    if torch.is_tensor(x):return x.numel()*x.element_size()
    if isinstance(x,(tuple,list)):return sum(tensor_bytes(v) for v in x)
    return 0
