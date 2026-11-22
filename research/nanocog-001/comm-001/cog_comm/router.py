"""Temporary causal COG routing reference; bounded scratch, no persistent memories."""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteState:
    values: Tensor         # [B,4,2,C], newest first
    positions: Tensor      # [B,4,2], absolute stream position, -1 when absent
    cursor: Tensor         # [B], position of next token

    def index_select(self, rows: Tensor):
        return RouteState(*(value.index_select(0, rows) for value in
                            (self.values, self.positions, self.cursor)))

    def detached(self):
        return RouteState(*(value.detach().clone() for value in
                            (self.values, self.positions, self.cursor)))


def _ste(hard, soft):
    return hard + (soft - soft.detach())


class BoundedRoute(nn.Module):
    """Per-layer delta. Read two topics with <=2 entries each; publish after read.

    Scratch covers a fixed number of recent aligned blocks. Routing
    uses the input representation, never the oracle support address.
    """
    TOPICS = 4
    ADMITTED_TOPICS = 2
    PER_TOPIC = 2

    def __init__(self, width: int, block: int = 8, *, retained_blocks: int = 16, route_mode: str = "learned"):
        super().__init__()
        if min(width, block, retained_blocks) < 1:
            raise ValueError("width and block must be positive")
        if route_mode not in ("learned","content_blind"):
            raise ValueError("unsupported routing mode")
        self.width, self.block = width, block
        self.retained_blocks = retained_blocks
        self.route_mode = route_mode
        self.publisher = nn.Linear(width, self.TOPICS, bias=False)
        self.subscriber = nn.Linear(width, self.TOPICS, bias=False)
        self.payload = nn.Linear(width, width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.lag = nn.Parameter(torch.zeros(self.PER_TOPIC))
        self.gain = nn.Linear(width, 1)
        self.admit = nn.Linear(width, 1)
        nn.init.constant_(self.admit.bias,0.15)

    @staticmethod
    def _linear(layer, x):
        return F.linear(x, layer.weight.to(dtype=x.dtype),
                        None if layer.bias is None else layer.bias.to(dtype=x.dtype))

    def initial_state(self, x):
        batch = x.shape[0]
        return RouteState(x.new_zeros(batch, self.TOPICS, self.PER_TOPIC, self.width),
                          torch.full((batch,self.TOPICS,self.PER_TOPIC),-1,
                                     device=x.device,dtype=torch.long),
                          torch.zeros(batch,device=x.device,dtype=torch.long))

    def _validate(self, x, state):
        if x.ndim != 3 or not x.shape[0] or x.shape[-1] != self.width or not x.is_floating_point():
            raise ValueError("expected nonempty floating B,T,C")
        if state is None:
            return self.initial_state(x)
        b = x.shape[0]
        if ((state.values.shape != (b, self.TOPICS, self.PER_TOPIC, self.width))
            or (state.positions.shape != (b, self.TOPICS, self.PER_TOPIC))
            or (state.cursor.shape != (b,))
            or state.values.dtype != x.dtype
            or state.positions.dtype != torch.long or state.cursor.dtype != torch.long
            or any(a.device != x.device for a in (state.values,state.positions,state.cursor))):
            raise ValueError("incompatible routing state")
        return state

    def _live(self, indices, cursor):
        now = cursor[:,None,None]
        return ((indices >= 0) & (indices < now)
                & (indices.div(self.block, rounding_mode="floor")
                   >= now.div(self.block, rounding_mode="floor") - (self.retained_blocks-1)))

    def _age_pool(self, payload, present):
        fp = torch.float64 if payload.dtype==torch.float64 else torch.float32
        age = self.lag.to(fp).softmax(-1)
        norm = (present.to(fp)*age).sum(-1,keepdim=True)
        numerator = (torch.where(present[...,None],payload,0).to(fp)*age[:,None]).sum(-2)
        return (numerator / torch.where(norm>0,norm,torch.ones_like(norm))).to(payload.dtype)

    def _run(self,x,state,disable,diagnostics):
        state = self._validate(x,state)
        b,t,c = x.shape
        values,idx,cursor = state.values,state.positions,state.cursor
        if disable:
            cursor = cursor+t
            live = self._live(idx,cursor)
            return torch.zeros_like(x),RouteState(torch.where(live[...,None],values,0),
                    torch.where(live,idx,-1),cursor),(
                    torch.full((b,t,2,2),-1,device=x.device,dtype=torch.long) if diagnostics else None)
        read = self._linear(self.subscriber,x)
        write = self._linear(self.publisher,x)
        messages = self._linear(self.payload,x)
        admission = self._linear(self.admit,x).squeeze(-1)
        groups = torch.argsort(read,dim=-1,descending=True,stable=True)[...,:self.ADMITTED_TOPICS]
        writers = write.argmax(-1)
        rows = torch.arange(b,device=x.device)[:,None]
        outs=[];hits=[]
        for i in range(t):
            learned=self.route_mode=="learned"
            if learned:
                ri=groups[:,i]
            else:
                turn=cursor.remainder(self.TOPICS)
                ri=torch.stack((turn,(turn+1).remainder(self.TOPICS)),dim=-1)
            wi=writers[:,i]
            admitted=admission[:,i]>=0
            write_hard=F.one_hot(wi,self.TOPICS).to(x.dtype)
            live=self._live(idx,cursor)
            chosen_live=live[rows,ri]
            if diagnostics:
                hits.append(torch.where(chosen_live,idx[rows,ri],-1))
            pair_w=(read[:,i].gather(-1,ri).softmax(-1) if learned
                    else x.new_full(ri.shape,1/self.ADMITTED_TOPICS))
            if self.training and learned:
                # Hard forward selection; fixed four-topic surrogate only.
                # Normalize separately after masking top1 for the other subscription.
                one=F.one_hot(ri,self.TOPICS).to(x.dtype)
                rr=read[:,i,None,:].expand_as(one)
                mask=torch.stack((torch.zeros_like(one[:,0],dtype=torch.bool),one[:,0].bool()),dim=1)
                soft=rr.masked_fill(mask,-torch.inf).softmax(-1)
                pooled=self._age_pool(values,live)
                mix=(_ste(one,soft).unsqueeze(-1)*pooled[:,None]).sum(dim=2)
            else:
                mix=self._age_pool(values[rows,ri],chosen_live)
            if self.training:
                wp=_ste(write_hard,write[:,i].softmax(-1))
                commit=_ste(admitted.to(x.dtype),admission[:,i].sigmoid())
            else:
                wp=write_hard
                commit=admitted.to(x.dtype)
            wp=wp*commit[:,None]
            outs.append((mix*pair_w[...,None]).sum(1))
            # Current h may affect addresses and values, never appears as a read candidate.
            repl=torch.cat((messages[:,i,None,None,:].expand(b,self.TOPICS,1,c),
                            values[:,:,:-1]),dim=2)
            repl_idx=torch.cat((cursor[:,None,None].expand(b,self.TOPICS,1),idx[:,:,:-1]),dim=2)
            values=(1-wp[:,:,None,None])*values+wp[:,:,None,None]*repl
            idx=torch.where((write_hard.bool() & admitted[:,None])[...,None],repl_idx,idx)
            cursor=cursor+1
            nxt=self._live(idx,cursor)
            values=torch.where(nxt[...,None],values,0)
            idx=torch.where(nxt,idx,-1)
        mixed=torch.stack(outs,dim=1) if outs else x
        gain=self._linear(self.gain,x).sigmoid()
        delta=self._linear(self.project,mixed)*gain
        indices=(torch.stack(hits,dim=1) if hits else
                 torch.empty((b,0,2,2),device=x.device,dtype=torch.long)) if diagnostics else None
        return delta,RouteState(values,idx,cursor),indices

    def forward(self,x,state=None,*,disable=False):
        delta,nxt,_=self._run(x,state,disable,False)
        return delta,nxt

    def forward_with_candidates(self,x,state=None,*,disable=False):
        """Same forward; candidate positions shaped [B,T,2,2] (-1 absent)."""
        return self._run(x,state,disable,True)
