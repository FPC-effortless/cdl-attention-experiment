"""Phase-1 Static Mask and CASM-S models.

CASM-S uses only structural inputs for routing; runtime values never enter the
router. alpha is relation-indexed by syntactic argument port, never by edge.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .grammar import Op

OP_IDS = {Op.INPUT: 0, Op.NOT: 1, Op.AND: 2, Op.OR: 3, Op.XOR: 4}

class _Base(nn.Module):
    def __init__(self, max_nodes=10, dim=32, temperature=2.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.max_nodes = max_nodes
        self.temperature = temperature
        self.op_emb = nn.Embedding(5, dim)
        self.depth_proj = nn.Linear(1, dim, bias=False)
        self.pos_proj = nn.Linear(1, dim, bias=False)
        self.msg = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        # Exactly one learned strength per syntactic relation/argument port.
        self.alpha_eta = nn.Parameter(torch.zeros(2))
        self.register_buffer("c", torch.tensor(1.0))

    def structural_encode(self, episodes):
        B, N = len(episodes), self.max_nodes
        device = self.op_emb.weight.device
        ops = torch.zeros(B, N, dtype=torch.long, device=device)
        depth = torch.zeros(B, N, 1, device=device)
        pos = torch.zeros(B, N, 1, device=device)
        exist = torch.zeros(B, N, 1, device=device)
        for b, ep in enumerate(episodes):
            for node in ep.nodes:
                ops[b, node.index] = OP_IDS[node.op]
                depth[b, node.index, 0] = node.depth / max(1, N)
                pos[b, node.index, 0] = node.index / max(1, N - 1)
            exist[b, :ep.active_count, 0] = 1.0
        h = self.op_emb(ops) + self.depth_proj(depth) + self.pos_proj(pos)
        # Structural message passing uses true program wiring only. It is fed
        # before runtime execution and is therefore independent of X_runtime.
        for b, ep in enumerate(episodes):
            acc = torch.zeros_like(h[b])
            for e in ep.true_edges:
                acc[e.dst] += self.msg(h[b, e.src])
            h[b] = self.norm(h[b] + acc)
        return h * exist, exist.squeeze(-1)

    def _edge_tensors(self, episodes, device):
        E = max(len(ep.candidate_edges) for ep in episodes)
        src = torch.zeros(len(episodes), E, dtype=torch.long, device=device)
        dst = torch.zeros_like(src); port = torch.zeros_like(src)
        valid = torch.zeros(len(episodes), E, device=device)
        truth = torch.zeros_like(valid)
        for b, ep in enumerate(episodes):
            for k, e in enumerate(ep.candidate_edges):
                src[b,k], dst[b,k], port[b,k] = e.src, e.dst, e.port
                valid[b,k] = 1.0
                truth[b,k] = float((e.src,e.dst,e.port) in ep.true_edge_set)
        return src, dst, port, valid, truth

    def gate(self, episodes):
        raise NotImplementedError

    def forward(self, episodes, runtime_inputs):
        h, exist = self.structural_encode(episodes)
        gates, edge_meta = self.gate(episodes)
        src, dst, port, valid, _ = edge_meta
        B, N = h.shape[:2]
        values = torch.zeros(B, N, device=h.device)
        for b, ep in enumerate(episodes):
            values[b, list(ep.inputs)] = runtime_inputs[b, :len(ep.inputs)]
            for node in ep.nodes[:ep.active_count]:
                if node.op is Op.INPUT:
                    continue
                for p in range(node.arity):
                    mask = (dst[b] == node.index) & (port[b] == p) & (valid[b] > 0)
                    s = src[b, mask]
                    g = gates[b, mask]
                    if s.numel() == 0:
                        routed = torch.tensor(0.5, device=h.device)
                    else:
                        denom = g.sum() + 1e-6
                        routed = (g * values[b, s]).sum() / denom
                    if p == 0: a0 = routed
                    else: a1 = routed
                alpha = self.c * torch.nn.functional.softplus(self.alpha_eta)
                if node.arity == 1: x = alpha[0] * a0
                else: x0, x1 = alpha[0] * a0, alpha[1] * a1
                if node.op is Op.NOT: values[b,node.index] = 1 - x
                elif node.op is Op.AND: values[b,node.index] = x0 * x1
                elif node.op is Op.OR: values[b,node.index] = x0 + x1 - x0*x1
                elif node.op is Op.XOR: values[b,node.index] = x0 + x1 - 2*x0*x1
        return values[torch.arange(B, device=h.device), torch.tensor([e.output for e in episodes], device=h.device)], gates, edge_meta

class CASMS(_Base):
    def __init__(self, max_nodes=10, dim=32, temperature=2.0, seed=0):
        super().__init__(max_nodes, dim, temperature, seed)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.bias_relation = nn.Parameter(torch.zeros(2))

    def gate(self, episodes):
        h, _ = self.structural_encode(episodes)
        src, dst, port, valid, truth = self._edge_tensors(episodes, h.device)
        q = self.q(h); k = self.k(h)
        logits = (q.gather(1, dst[...,None].expand(-1,-1,h.size(-1))) *
                  k.gather(1, src[...,None].expand(-1,-1,h.size(-1)))).sum(-1) / math.sqrt(h.size(-1))
        logits = logits + self.bias_relation[port.clamp(max=1)]
        gates = torch.sigmoid(logits / self.temperature) * valid
        return gates, (src,dst,port,valid,truth)

class StaticMask(_Base):
    def __init__(self, max_nodes=10, dim=32, temperature=2.0, seed=0):
        super().__init__(max_nodes, dim, temperature, seed)
        self.edge_logits = nn.Parameter(torch.zeros(max_nodes, max_nodes, 2))

    def gate(self, episodes):
        dummy, exist = self.structural_encode(episodes)
        src, dst, port, valid, truth = self._edge_tensors(episodes, dummy.device)
        logits = self.edge_logits[dst, src, port.clamp(max=1)].to(dummy.device)
        gates = torch.sigmoid(logits / self.temperature) * valid
        return gates, (src,dst,port,valid,truth)


def copy_mask_gates(episodes, device=None):
    if device is None: device = torch.device("cpu")
    rows=[]
    for ep in episodes:
        true = ep.true_edge_set
        rows.append(torch.tensor([float((e.src,e.dst,e.port) in true) for e in ep.candidate_edges], device=device))
    return rows
