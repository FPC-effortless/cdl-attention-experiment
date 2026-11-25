"""Causal local mixing + learned bounded packet topics; independent PyTorch reference."""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass(frozen=True)
class PacketState:
    local: Tensor       # B x (W-1) x C, latest first
    packets: Tensor     # B x G x A x K x D, latest first
    present: Tensor     # B x G x A x K, hard binary forward; STE may carry gradients
    positions: Tensor   # B int64, valid tokens processed in this layer

    def _parts(self):
        return (self.local,self.packets,self.present,self.positions)

    def detached(self):
        """Detached independent storage for safe forks and resumable sessions."""
        return PacketState(*(a.detach().clone() for a in self._parts()))

    def index_select(self, rows: Tensor):
        if rows.ndim!=1 or rows.dtype!=torch.long or rows.device!=self.positions.device:
            raise ValueError("fork rows must be a one-dimensional int64 tensor on state device")
        return PacketState(*(x.index_select(0,rows) for x in self._parts()))

    def reset_rows(self, reset: Tensor):
        if (reset.shape!=self.positions.shape or reset.dtype!=torch.bool
                or reset.device!=self.positions.device):
            raise ValueError("reset must be a same-device Boolean batch vector")
        return PacketState(*(torch.where(reset.reshape(-1,*([1]*(a.ndim-1))),
                                          torch.zeros_like(a),a) for a in self._parts()))

    @property
    def nbytes(self):
        return sum(a.numel()*a.element_size() for a in self._parts())


def _ste(fwd: Tensor, bwd: Tensor):
    return fwd+(bwd-bwd.detach())


class PacketCommunication(nn.Module):
    """Per-layer mixer, no residual/norm. Hard routing in forward; ST training surrogate."""
    def __init__(self, width: int, groups: int=4, topics: int=8,
                 packet_width: int=16, capacity: int=4, local_window: int=4):
        super().__init__()
        if min(width,groups,topics,packet_width,capacity,local_window)<1:
            raise ValueError("sizes must be positive")
        self.width=width;self.groups=groups;self.topics=topics
        self.packet_width=packet_width;self.capacity=capacity;self.local_window=local_window
        self.local_age=nn.Parameter(torch.zeros(local_window,width))
        self.local_out=nn.Linear(width,width,bias=False)
        self.publish=nn.Linear(width,groups)
        self.write_topic=nn.Linear(width,groups*topics,bias=False)
        self.read_topic=nn.Linear(width,groups*topics,bias=False)
        self.payload=nn.Linear(width,groups*packet_width,bias=False)
        self.packet_out=nn.Linear(groups*packet_width,width,bias=False)
        self.packet_gain=nn.Linear(width,groups)
        self.packet_age=nn.Parameter(torch.zeros(groups,capacity))
        with torch.no_grad():
            self.local_age[0].fill_(1.)
            self.publish.bias.fill_(0.1)
            self.packet_gain.bias.fill_(-2.)

    @staticmethod
    def linear(m: nn.Linear, x: Tensor):
        return F.linear(x,m.weight.to(x.dtype),m.bias.to(x.dtype) if m.bias is not None else None)

    def initial_state(self, batch: int, *, device=None, dtype=None):
        if batch<1:
            raise ValueError("empty batch is not supported")
        w=self.local_age
        device=w.device if device is None else device
        dtype=w.dtype if dtype is None else dtype
        g,a,k,d=self.groups,self.topics,self.capacity,self.packet_width
        return PacketState(torch.zeros((batch,self.local_window-1,self.width),device=device,dtype=dtype),
                           torch.zeros((batch,g,a,k,d),device=device,dtype=dtype),
                           torch.zeros((batch,g,a,k),device=device,dtype=dtype),
                           torch.zeros(batch,device=device,dtype=torch.long))

    def _validate(self,x,state,valid):
        if x.ndim!=3 or x.shape[-1]!=self.width or not x.is_floating_point() or not x.shape[0]:
            raise ValueError("x must be a nonempty floating B,T,width tensor")
        b,t,_=x.shape
        if valid is None:
            valid=torch.ones((b,t),dtype=torch.bool,device=x.device)
        if valid.shape!=(b,t) or valid.dtype!=torch.bool or valid.device!=x.device:
            raise ValueError("valid must be a same-device Boolean B,T tensor")
        if state is None:
            state=self.initial_state(b,device=x.device,dtype=x.dtype)
        shapes=((b,self.local_window-1,self.width),(b,self.groups,self.topics,self.capacity,self.packet_width),
                (b,self.groups,self.topics,self.capacity),(b,))
        if any(y.shape!=s or y.device!=x.device or y.dtype!=dt for y,s,dt in
               zip(state._parts(),shapes,(x.dtype,x.dtype,x.dtype,torch.long),strict=True)):
            raise ValueError("packet state shape/device/dtype mismatch")
        return state,valid

    def _project(self,x):
        b,t,_=x.shape;g,a,d=self.groups,self.topics,self.packet_width
        return (self.linear(self.publish,x),self.linear(self.write_topic,x).view(b,t,g,a),
                self.linear(self.read_topic,x).view(b,t,g,a),self.linear(self.payload,x).view(b,t,g,d),
                self.linear(self.packet_gain,x).sigmoid())

    @staticmethod
    def _pool(packets,present,age):
        # packets already carry zero at empty slots in ordinary forward.
        # Accumulate normalization in FP32 for low precision; empty buckets are exactly zero.
        fp=torch.float64 if packets.dtype==torch.float64 else torch.float32
        den=(present.to(fp)*age).sum(dim=-1,keepdim=True)
        num=(packets.to(fp)*age.unsqueeze(-1)).sum(dim=-2)
        return (num/torch.where(den>0,den,torch.ones_like(den))).to(packets.dtype)

    def _core(self,x,state,valid,use_packets,owned,need_aux):
        b,t,c=x.shape;g,a,k,d=self.groups,self.topics,self.capacity,self.packet_width
        x=torch.where(valid.unsqueeze(-1),x,0)
        local,packs,present,pos=state._parts()
        age=self.packet_age.to(torch.float64 if x.dtype==torch.float64 else torch.float32).softmax(dim=-1)
        pub,wr,rd,vals,gain=self._project(x) if use_packets else (None,)*5
        features=[];pack_features=[]
        dense=self.training or torch.is_grad_enabled()
        if use_packets:
            rid=rd.argmax(dim=-1);wid=wr.argmax(dim=-1)
            active_publish=(pub>=0)&valid[:,:,None]
        if use_packets and not dense:
            row=torch.arange(b,device=x.device)[:,None]; group=torch.arange(g,device=x.device)[None,:]
            new_packs=packs if owned else packs.clone()
            new_present=present if owned else present.clone()
        for i in range(t):
            ok=valid[:,i];current=x[:,i]
            window=torch.cat((current[:,None,:],local),dim=1)
            features.append((window*self.local_age.to(x.dtype)).sum(dim=1))
            local=torch.where(ok[:,None,None],window[:,:self.local_window-1],local)
            if not use_packets:continue
            ri=rid[:,i];wi=wid[:,i]
            publish=active_publish[:,i]
            # Read strictly from old state, then publish. No current packet leakage.
            if dense:
                summaries=self._pool(packs,present,age[None,:,None,:])
                read_hard=F.one_hot(ri,a).to(x.dtype)
                wp=F.one_hot(wi,a).to(x.dtype)
                gate=publish.to(x.dtype)
                if self.training:
                    read_hard=_ste(read_hard,rd[:,i].softmax(dim=-1))
                    wp=_ste(wp,wr[:,i].softmax(dim=-1))
                    gate=_ste(gate,pub[:,i].sigmoid()*ok[:,None])
                msg=(read_hard.unsqueeze(-1)*summaries).sum(dim=2)
                alpha=(wp*gate.unsqueeze(-1)).unsqueeze(-1)
                replacement=torch.cat((vals[:,i,:,None,None,:].expand(b,g,a,1,d),packs[:,:,:,:-1]),dim=-2)
                replacement_present=torch.cat((torch.ones_like(present[:,:,:,:1]),present[:,:,:,:-1]),dim=-1)
                packs=(1-alpha.unsqueeze(-1))*packs+alpha.unsqueeze(-1)*replacement
                present=(1-alpha)*present+alpha*replacement_present
            else:
                old=new_packs[row,group,ri]
                old_present=new_present[row,group,ri]
                msg=self._pool(old,old_present,age[None,:,:])
                dest=new_packs[row,group,wi];dp=new_present[row,group,wi]
                next_dest=torch.cat((vals[:,i,:,None,:],dest[:,:,:-1]),dim=-2)
                next_dp=torch.cat((torch.ones_like(dp[:,:,:1]),dp[:,:,:-1]),dim=-1)
                new_packs[row,group,wi]=torch.where(publish[:,:,None,None],next_dest,dest)
                new_present[row,group,wi]=torch.where(publish[:,:,None],next_dp,dp)
            pack_features.append((msg*gain[:,i].unsqueeze(-1)).reshape(b,g*d))
        local_out=self.linear(self.local_out,torch.stack(features,dim=1) if features else x)
        if use_packets and pack_features:
            local_out=local_out+self.linear(self.packet_out,torch.stack(pack_features,dim=1))
        out=torch.where(valid.unsqueeze(-1),local_out,0)
        if use_packets and not dense:
            packs,present=new_packs,new_present
        if not use_packets and not owned:
            packs,present=packs.clone(),present.clone()
        next_pos=pos+valid.sum(dim=1)
        if owned:
            # Inference-only fast path updates caller-owned storage; functional path does not.
            state.local.copy_(local);state.positions.copy_(next_pos)
            updated=state
        else:
            updated=PacketState(local,packs,present,next_pos)
            if t == 0:
                updated=PacketState(*(tensor.clone() for tensor in updated._parts()))
        aux={} if not (use_packets and need_aux) else {
            'publish_logits':pub,'write_topic_logits':wr,'read_topic_logits':rd,
            'published':active_publish,
            'write_topic':wid,'read_topic':rid,
            'valid':valid,
        }
        return out,updated,aux

    def forward(self,x,state=None,*,valid=None,return_state=False,return_aux=False,use_packets=True):
        """Functional chunks: outputs, optional next state, optional route diagnostics."""
        state,valid=self._validate(x,state,valid)
        out,new,aux=self._core(x,state,valid,use_packets,False,return_aux)
        if return_state and return_aux:return out,new,aux
        if return_state:return out,new
        if return_aux:return out,aux
        return out

    def step_inplace_(self,x,state,*,valid=None,use_packets=True):
        """Decode-only, explicitly owned state. Caller must fork with index_select."""
        if self.training or not torch.is_inference_mode_enabled():
            raise RuntimeError("in-place decode requires eval and torch.inference_mode()")
        state,valid=self._validate(x,state,valid)
        return self._core(x,state,valid,use_packets,True,False)[0]
