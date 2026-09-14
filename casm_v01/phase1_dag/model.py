"""Phase-1 Static Mask and CASM-S models.

CASM-S routes only from non-oracle structural descriptors; runtime values and
true program edges never enter the router. alpha is relation-indexed by
syntactic argument port, never by physical edge.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .grammar import Op

OP_IDS={Op.INPUT:0,Op.NOT:1,Op.AND:2,Op.OR:3,Op.XOR:4}

class _Base(nn.Module):
    def __init__(self,max_nodes=10,dim=32,temperature=2.0,seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.max_nodes=max_nodes; self.temperature=temperature
        self.op_emb=nn.Embedding(5,dim); self.depth_proj=nn.Linear(1,dim,bias=False); self.pos_proj=nn.Linear(1,dim,bias=False)
        self.arity_proj=nn.Linear(1,dim,bias=False); self.norm=nn.LayerNorm(dim)
        self.alpha_eta=nn.Parameter(torch.zeros(2)); self.register_buffer("c",torch.tensor(1.0))
        self.last_node_values=None

    def structural_encode(self,episodes):
        """Encode only public per-node structure, never the oracle edge set."""
        B,N=len(episodes),self.max_nodes; device=self.op_emb.weight.device
        ops=torch.zeros(B,N,dtype=torch.long,device=device); depth=torch.zeros(B,N,1,device=device); pos=torch.zeros(B,N,1,device=device); arity=torch.zeros(B,N,1,device=device); exist=torch.zeros(B,N,1,device=device)
        for b,ep in enumerate(episodes):
            for node in ep.nodes[:ep.active_count]:
                ops[b,node.index]=OP_IDS[node.op]; depth[b,node.index,0]=node.depth/max(1,N); pos[b,node.index,0]=node.index/max(1,N-1); arity[b,node.index,0]=node.arity/2
            exist[b,:ep.active_count,0]=1
        h=self.op_emb(ops)+self.depth_proj(depth)+self.pos_proj(pos)+self.arity_proj(arity)
        return self.norm(h)*exist,exist.squeeze(-1)

    def _edge_tensors(self,episodes,device):
        E=max(len(ep.candidate_edges) for ep in episodes); B=len(episodes)
        src=torch.zeros(B,E,dtype=torch.long,device=device); dst=torch.zeros_like(src); port=torch.zeros_like(src); valid=torch.zeros(B,E,device=device); truth=torch.zeros_like(valid)
        for b,ep in enumerate(episodes):
            for k,e in enumerate(ep.candidate_edges):
                src[b,k],dst[b,k],port[b,k]=e.src,e.dst,e.port; valid[b,k]=1; truth[b,k]=float((e.src,e.dst,e.port) in ep.true_edge_set)
        return src,dst,port,valid,truth

    def gate(self,episodes): raise NotImplementedError

    def forward(self,episodes,runtime_inputs):
        h,_=self.structural_encode(episodes); gates,meta=self.gate(episodes); src,dst,port,valid,_=meta
        B,N=h.shape[:2]
        values=[torch.zeros(B,device=h.device) for _ in range(N)]
        zero=torch.zeros(B,device=h.device)
        for b,ep in enumerate(episodes):
            for j,node_idx in enumerate(ep.inputs):
                mask=torch.zeros(B,dtype=torch.bool,device=h.device); mask[b]=True
                values[node_idx]=torch.where(mask,runtime_inputs[:,j],values[node_idx])
        alpha=self.c*torch.nn.functional.softplus(self.alpha_eta)
        for node_idx in range(N):
            for b,ep in enumerate(episodes):
                if node_idx>=ep.active_count: continue
                node=ep.nodes[node_idx]
                if node.op is Op.INPUT: continue
                args=[]
                for p in range(node.arity):
                    mask=(dst[b]==node.index)&(port[b]==p)&(valid[b]>0); s=src[b,mask]; g=gates[b,mask]
                    if s.numel()==0: routed=zero[b]
                    else:
                        routed=(g*torch.stack([values[int(si)][b] for si in s])).sum()/(g.sum()+1e-6)
                    args.append(routed)
                if node.op is Op.NOT: out=1-alpha[0]*args[0]
                else:
                    a,bv=alpha[0]*args[0],alpha[1]*args[1]
                    if node.op is Op.AND: out=a*bv
                    elif node.op is Op.OR: out=a+bv-a*bv
                    elif node.op is Op.XOR: out=a+bv-2*a*bv
                    else: out=zero[b]
                values[node.index]=torch.cat([values[node.index][:b],out.view(1),values[node.index][b+1:]])
        node_matrix=torch.stack(values,dim=1)
        self.last_node_values=node_matrix
        idx=torch.tensor([e.output for e in episodes],device=h.device)
        return node_matrix[torch.arange(B,device=h.device),idx],gates,meta

class CASMS(_Base):
    def __init__(self,max_nodes=10,dim=32,temperature=2.0,seed=0):
        super().__init__(max_nodes,dim,temperature,seed); self.q=nn.Linear(dim,dim,bias=False); self.k=nn.Linear(dim,dim,bias=False); self.bias_relation=nn.Parameter(torch.zeros(2))
    def gate(self,episodes):
        h,_=self.structural_encode(episodes); src,dst,port,valid,truth=self._edge_tensors(episodes,h.device); q=self.q(h); k=self.k(h)
        qd=q.gather(1,dst[...,None].expand(-1,-1,h.size(-1))); ks=k.gather(1,src[...,None].expand(-1,-1,h.size(-1)))
        logits=(qd*ks).sum(-1)/math.sqrt(h.size(-1))+self.bias_relation[port.clamp(max=1)]
        return torch.sigmoid(logits/self.temperature)*valid,(src,dst,port,valid,truth)

class StaticMask(_Base):
    def __init__(self,max_nodes=10,dim=32,temperature=2.0,seed=0):
        super().__init__(max_nodes,dim,temperature,seed); self.edge_logits=nn.Parameter(torch.zeros(max_nodes,max_nodes,2))
    def gate(self,episodes):
        h,_=self.structural_encode(episodes); src,dst,port,valid,truth=self._edge_tensors(episodes,h.device)
        logits=self.edge_logits[dst,src,port.clamp(max=1)]; return torch.sigmoid(logits/self.temperature)*valid,(src,dst,port,valid,truth)

def copy_mask_gates(episodes,device=None):
    device=device or torch.device("cpu"); return [torch.tensor([float((e.src,e.dst,e.port) in ep.true_edge_set) for e in ep.candidate_edges],device=device) for ep in episodes]
