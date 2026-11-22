"""Functional, typed, fixed-address state and a small learned routing adapter."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

HOLD, SET, CLEAR = 0, 1, 2


@dataclass(frozen=True)
class RecordState:
    value: torch.Tensor       # int64 [B,R], -1 is empty
    revision: torch.Tensor    # int64 [B,R], counts material changes only
    clock: torch.Tensor       # int64 [B], visible sequence step count
    last_change: torch.Tensor # int64 [B,R], -1 for never

    def detach(self):
        return RecordState(*(x.detach().clone() for x in self._parts()))

    def index_select(self, ix):
        return RecordState(*(x.index_select(0,ix) for x in self._parts()))

    def _parts(self):
        return (self.value,self.revision,self.clock,self.last_change)


@dataclass(frozen=True)
class WritePacket:
    address: torch.Tensor     # int64 B
    operation: torch.Tensor   # int64 B, HOLD | SET | CLEAR
    value: torch.Tensor       # int64 B, used only for SET


@dataclass(frozen=True)
class ApplyResult:
    legal: torch.Tensor       # bool B, legal active packet (including HOLD)
    changed: torch.Tensor     # bool B, published a new material revision


@dataclass(frozen=True)
class StateRead:
    address: torch.Tensor
    value: torch.Tensor       # -1 if unavailable or invalid; may be a typed link
    resolved: torch.Tensor    # -1 if unavailable; after at most one typed link traversal
    target: torch.Tensor      # stable final record address, or -1
    valid: torch.Tensor


class TypedRecordBoard(nn.Module):
    """Stateless interpreter for one batched atomic packet per visible event.

    Ontology is a fixed public model interface, not an annotated read or write
    address. Default IDs follow PULSE006: addresses BOX+CARGO; VALUE=NAMES+BOX.
    """
    def __init__(self, permits=None, links=None):
        super().__init__()
        if permits is None:
            permits=torch.zeros((8,12),dtype=torch.bool)
            permits[:4,:8]=True
            permits[4:,8:]=True
            if links is None:
                links=torch.tensor([-1]*8+[0,1,2,3],dtype=torch.long)
        if (not torch.is_tensor(permits) or permits.dtype!=torch.bool or permits.ndim!=2 or
                min(permits.shape)<1 or not permits.any(1).all()):
            raise ValueError("expected nonempty bool [record,value] schema")
        if links is None:
            links=torch.full((permits.size(1),),-1,dtype=torch.long,device=permits.device)
        if (not torch.is_tensor(links) or links.dtype!=torch.long or links.shape!=(permits.size(1),)
                or links.device!=permits.device or (links< -1).any() or (links>=permits.size(0)).any()):
            raise ValueError("invalid value-to-stable-address schema")
        self.register_buffer('permits',permits.detach().clone())
        self.register_buffer('links',links.detach().clone())

    @property
    def records(self): return self.permits.shape[0]

    @property
    def values(self): return self.permits.shape[1]

    def empty(self,batch,device=None):
        device=self.permits.device if device is None else torch.device(device)
        if device!=self.permits.device:raise ValueError('board and state must share device')
        shape=(batch,self.records)
        return RecordState(torch.full(shape,-1,device=device,dtype=torch.long),
                           torch.zeros(shape,device=device,dtype=torch.long),
                           torch.zeros(batch,device=device,dtype=torch.long),
                           torch.full(shape,-1,device=device,dtype=torch.long))

    def _shape(self,state,x,active=None):
        if not isinstance(state,RecordState):raise TypeError("use a RecordState")
        if state.value.ndim!=2 or state.value.size(1)!=self.records:
            raise ValueError("invalid board state shape")
        b=state.value.size(0); shape=(b,self.records)
        want=(shape,shape,(b,),shape)
        for t,s in zip(state._parts(),want):
            if t.dtype!=torch.long or t.shape!=s or t.device!=self.permits.device:
                raise ValueError("state/board shape, dtype or device mismatch")
        for t in x:
            if not torch.is_tensor(t) or t.dtype!=torch.long or t.shape!=(b,) or t.device!=state.value.device:
                raise ValueError("packet field must be int64 [batch] on state device")
        if active is not None and (not torch.is_tensor(active) or active.dtype!=torch.bool
                                  or active.shape!=(b,) or active.device!=state.value.device):
            raise ValueError("activity must be bool [batch] on state device")
        return b

    def validate_state(self,state):
        """Explicit trust check at a persistence/import boundary, not in hot loop."""
        self._shape(state,())
        val=state.value; addr=torch.arange(self.records,device=val.device)[None,:]
        occupied=val>=0
        legal=self.permits[addr,val.clamp(0,self.values-1)]
        if ((val< -1)|(val>=self.values)|(occupied&~legal)).any():raise ValueError("invalid typed stored value")
        if (state.clock<0).any() or (state.revision<0).any():raise ValueError("negative chronology")
        if (((state.revision==0)!=(state.last_change==-1))|(state.last_change< -1)|
                (state.last_change>=state.clock[:,None])|(state.revision>state.clock[:,None])|
                (occupied&(state.revision==0))).any():
            raise ValueError("contradictory record chronology")
        # At most one material change per consumed step. Sorted final-change
        # deadlines provide a cheap necessary check, not a replay or authenticity proof.
        deadlines, order = state.last_change.sort(dim=1)
        revisions = state.revision.gather(1, order)
        if (revisions.cumsum(dim=1) > deadlines + 1).any():
            raise ValueError("contradictory aggregate record chronology")
        if ((deadlines[:, 1:] >= 0) & (deadlines[:, 1:] == deadlines[:, :-1])).any():
            raise ValueError("duplicate material change at one event")
        return state

    def apply(self,state,packet,active=None):
        """Advance consumed live events. A rejected packet cannot change any row."""
        self._shape(state,(packet.address,packet.operation,packet.value),active)
        val=state.value; a,op,v=packet.address,packet.operation,packet.value
        if active is None:active=torch.ones_like(a,dtype=torch.bool)
        ac=a.clamp(0,self.records-1); vc=v.clamp(0,self.values-1)
        avalid=(a>=0)&(a<self.records)
        writable=(op==SET)&(v>=0)&(v<self.values)&self.permits[ac,vc]
        legal=active&((op==HOLD)|(avalid&(writable|(op==CLEAR))))
        new=torch.where(op==CLEAR,torch.full_like(v,-1),v)
        old=val.gather(1,ac[:,None])[:,0]
        changed=legal&(op!=HOLD)&(new!=old)
        update=changed[:,None] & F.one_hot(ac,self.records).bool()
        return RecordState(torch.where(update,new[:,None],val),
                           state.revision+update.long(),
                           state.clock+active.long(),
                           torch.where(update,state.clock[:,None],state.last_change)),ApplyResult(legal,changed)

    def read(self,state,address,active=None):
        self._shape(state,(address,),active)
        a=address.clamp(0,self.records-1)
        seen=state.value.gather(1,a[:,None])[:,0]
        empty=torch.full_like(seen,-1)
        good=(address>=0)&(address<self.records)&(seen>=0)&(seen<self.values)
        good &= self.permits[a,seen.clamp(0,self.values-1)]
        if active is not None:good &= active
        link=self.links[seen.clamp(0,self.values-1)]
        link_addr=link.clamp(0,self.records-1)
        second=state.value.gather(1,link_addr[:,None])[:,0]
        second_ok=(second>=0)&(second<self.values)
        second_ok &= self.permits[link_addr,second.clamp(0,self.values-1)]
        second_ok &= self.links[second.clamp(0,self.values-1)]<0
        resolved_valid=good & ((link<0)|second_ok)
        target=torch.where(link<0,a,link_addr)
        return StateRead(address,torch.where(good,seen,empty),
                         torch.where(resolved_valid,torch.where(link<0,seen,second),empty),
                         torch.where(resolved_valid,target,empty),resolved_valid)


class StableRecordMemory(nn.Module):
    """Raw B,T,C hidden events → learned proposals → causal bounded records.

    Hard address/value/operation choices intentionally do not carry answer-loss
    gradient across the packet. Train them only with a prospective, matched
    estimator or explicit supervision (aux returns raw logits). The output graph
    learns payload embedding and readout without claiming end-to-end write learning.
    """
    def __init__(self,width,permits=None,links=None,key_width=16):
        super().__init__()
        if width<1 or key_width<1:raise ValueError("positive widths required")
        self.width=width
        self.board=TypedRecordBoard(permits,links)
        self.key=nn.Parameter(torch.empty(self.board.records,key_width))
        nn.init.normal_(self.key,std=key_width**-.5)
        self.write_address=nn.Linear(width,key_width,bias=False)
        self.read_address=nn.Linear(width,key_width,bias=False)
        self.write_operation=nn.Linear(width,3)
        self.write_value=nn.Linear(width,self.board.values)
        self.payload=nn.Embedding(self.board.values,width)
        self.output=nn.Linear(width,width,bias=False)

    def forward(self,x,state=None,*,valid=None,return_state=False,return_aux=False):
        if x.ndim!=3 or x.shape[-1]!=self.width:
            raise ValueError("expected [batch,time,width]")
        b,t,_=x.shape
        if valid is None:valid=torch.ones((b,t),device=x.device,dtype=torch.bool)
        if valid.shape!=(b,t) or valid.dtype!=torch.bool or valid.device!=x.device:
            raise ValueError("expected bool [batch,time] mask")
        if state is None:state=self.board.empty(b,x.device)
        if state.value.size(0)!=b:raise ValueError("batch/state mismatch")
        safe=x.masked_fill(~valid[:,:,None],0)
        r=self.read_address(safe) @ self.key.T
        a=self.write_address(safe) @ self.key.T
        op=self.write_operation(safe);v=self.write_value(safe)
        outs=[];traces=[]
        for i in range(t):
            live=valid[:,i]
            read=self.board.read(state,r[:,i].argmax(-1),live)
            y=self.output(self.payload(read.resolved.clamp(0,self.board.values-1)))
            outs.append(y.masked_fill(~read.valid[:,None],0))
            addr=a[:,i].argmax(-1)
            selection=v[:,i].masked_fill(~self.board.permits[addr],-torch.inf).argmax(-1)
            change=WritePacket(addr,op[:,i].argmax(-1),selection)
            state,receipt=self.board.apply(state,change,live)
            if return_aux:traces.append((read.resolved,read.valid,receipt.legal,receipt.changed))
        y=torch.stack(outs,dim=1) if outs else x.new_zeros((b,0,self.width))
        pack=(y,)
        if return_state:pack+=(state,)
        if return_aux:
            block=[torch.stack(col,1) for col in zip(*traces)] if traces else [
                torch.empty((b,0),dtype=d,device=x.device) for d in
                (torch.long,torch.bool,torch.bool,torch.bool)]
            pack+=({"read_logits":r,"address_logits":a,"operation_logits":op,"value_logits":v,
                    **dict(zip(("resolved","read_valid","write_legal","write_changed"),block))},)
        return pack[0] if len(pack)==1 else pack
