"""Adaptive side computations on the two exact Native HH layers from TRACE-007.

Add research/unified-search-002 to PYTHONPATH when loading outside the repository.
Multi-token training/prefill uses dense surrogate computation. Single-token
evaluation dispatches optional branches on selected rows only.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-search-002"))
from integrated import WholeCOG
from model import D, count, delta, hard, scan


def binary(logit):
    prob = logit.sigmoid()
    return (prob >= 0.5).to(prob.dtype) + (prob - prob.detach())


def affine_scan(matrix, initial):
    outputs = []
    for block in matrix.split(32, dim=1):
        shift = 1
        while shift < block.shape[1]:
            block = torch.cat((block[:, :shift], block[:, shift:] @ block[:, :-shift]), 1)
            shift *= 2
        out = (block @ initial[:, None, :, None]).squeeze(-1)
        outputs.append(out)
        initial = out[:, -1]
    return torch.cat(outputs, dim=1), initial


class AdaptiveSide(nn.Module):
    width = 8

    def __init__(self, slow, arm):
        super().__init__()
        self.slow = slow
        self.arm = arm
        self.context = nn.Linear(D, 8)
        self.key_query_value = nn.Linear(8, 24)
        self.rate = nn.Linear(8, 1)
        self.memory_out = nn.Linear(8, D, bias=False)
        nn.init.normal_(self.memory_out.weight, std=0.01)
        if arm >= "E":
            self.event = nn.Linear(8, 2)
        if arm >= "F":
            self.router = nn.Linear(8, 4 if arm == "H" else 3)
        if arm >= "G":
            self.reliability = nn.Linear(8, 1)
        if arm == "H":
            self.axes = nn.Parameter(torch.randn(4, 8))
            self.initial = nn.Parameter(torch.randn(8) * 0.1)
            self.axis_choice = nn.Linear(8, 4)
            self.geo_out = nn.Linear(8, D, bias=False)
            nn.init.normal_(self.geo_out.weight, std=0.01)
        self.measure = False
        self.stats = {}
        self.soft_cost = None

    def geometry(self, ctx, prev, active):
        axes = F.normalize(self.axes, dim=-1)
        # Identity or one of four independent Householder generators.
        p = hard(self.axis_choice(ctx).softmax(-1)) * active
        outer = axes[:, :, None] * axes[:, None, :]
        eye = torch.eye(8, dtype=ctx.dtype, device=ctx.device)
        op = eye - 2 * torch.einsum("bta,aij->btij", p, outer)
        return affine_scan(op, prev)

    def forward(self, h, state=None):
        b, t = h.shape[:2]
        prev_ctx = None if state is None else state[0]
        slow = h.new_zeros(b, 8) if state is None else state[1]
        memory = h.new_zeros(b, 8, 8) if state is None else state[2]
        geom = None
        if self.arm == "H":
            geom = self.initial.expand(b, -1) if state is None else state[3]
        u = self.slow.norm(h)
        value = self.context(u).tanh()
        # Context for addressing extends beyond Native's five-symbol local window.
        ctx, ctx_end = scan(value, torch.full_like(value, 0.75), prev_ctx)
        ones = torch.ones_like(ctx[..., :1])
        run_slow = run_mem = ones
        run_geo = torch.zeros_like(ones)
        if self.arm >= "F":
            probs = self.router(ctx).softmax(-1)
            choice = hard(probs)
            run_slow, run_mem = choice[..., 1:2], choice[..., 2:3]
            run_geo = choice[..., 3:4] if self.arm == "H" else run_geo
            self.soft_cost = (1 - probs[..., 0]).mean()
        permit_slow = permit_mem = ones
        if self.arm >= "E":
            permit_slow, permit_mem = binary(self.event(ctx)).chunk(2, -1)
        update_slow = permit_slow * run_slow
        update_mem = permit_mem * run_mem
        if self.measure:
            self.stats = {
                "positions": b * t,
                "slow_selected": run_slow.detach().sum(),
                "memory_selected": run_mem.detach().sum(),
                "geometric_selected": run_geo.detach().sum(),
                "slow_updates": update_slow.detach().sum(),
                "memory_writes": update_mem.detach().sum(),
            }
        if t == 1 and not self.training and self.arm >= "F":
            y, slow, memory, geom = self.decode_one(
                u[:, 0], ctx[:, 0], slow, memory, geom, run_slow[:, 0, 0].bool(),
                run_mem[:, 0, 0].bool(), run_geo[:, 0, 0].bool(),
                permit_slow[:, 0], permit_mem[:, 0],
            )
        else:
            z, a = self.slow.inp(u).chunk(2, -1)
            a = (a + self.slow.retention).sigmoid()
            slow_seq, slow = scan(z.tanh(), 1 - update_slow * (1 - a), slow)
            k, q, v = self.key_query_value(ctx).chunk(3, -1)
            k, q = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
            rate = self.rate(ctx).sigmoid() * update_mem
            if self.arm >= "G":
                rate = rate * self.reliability(ctx).sigmoid()
            values, memory = delta(k, v.tanh(), q, rate.squeeze(-1), memory)
            y = run_slow * self.slow.proj(slow_seq) + run_mem * self.memory_out(values)
            if self.arm == "H":
                geo_seq, geom = self.geometry(ctx, geom, run_geo)
                y = y + run_geo * self.geo_out(geo_seq)
        return y, (ctx_end, slow, memory, geom)

    def decode_one(self, u, ctx, slow, memory, geom, use_s, use_m, use_g, gate_s, gate_m):
        y = u.new_zeros(u.shape)
        if use_s.any():
            z, a = self.slow.inp(u[use_s]).chunk(2, -1)
            a = (a + self.slow.retention).sigmoid()
            step = gate_s[use_s] * (1 - a)
            new = slow[use_s] + step * (z.tanh() - slow[use_s])
            slow = slow.index_copy(0, use_s.nonzero().flatten(), new)
            y[use_s] = self.slow.proj(new)
        if use_m.any():
            c = ctx[use_m]
            k, q, v = self.key_query_value(c).chunk(3, -1)
            k, q = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
            rate = self.rate(c).sigmoid() * gate_m[use_m]
            if self.arm >= "G":
                rate = rate * self.reliability(c).sigmoid()
            mat = memory[use_m]
            error = v.tanh() - (k[:, None] @ mat).squeeze(1)
            new = mat + (rate * k).unsqueeze(-1) * error.unsqueeze(-2)
            memory = memory.index_copy(0, use_m.nonzero().flatten(), new)
            y[use_m] = self.memory_out((q[:, None] @ new).squeeze(1))
        if self.arm == "H" and use_g.any():
            ids = self.axis_choice(ctx[use_g]).argmax(-1)
            axis = F.normalize(self.axes[ids], dim=-1)
            prev = geom[use_g]
            new = prev - 2 * (prev * axis).sum(-1, keepdim=True) * axis
            geom = geom.index_copy(0, use_g.nonzero().flatten(), new)
            y[use_g] = self.geo_out(new)
        return y.unsqueeze(1), slow, memory, geom


class AdaptiveCOG(WholeCOG):
    def __init__(self, arm, seed):
        super().__init__("SS_native_native", seed)
        base_count = count(self)
        torch.manual_seed(seed + 63043)
        self.parent.sides = nn.ModuleList(AdaptiveSide(s, arm) for s in self.parent.sides)
        # Trade FF width for optional computation at fixed parameter budget.
        trim = round((count(self) - base_count) / (2 * (2 * D + 1)))
        self.parent.ff = tuple(width - trim for width in self.parent.ff)
        for block, width in zip(self.parent.core.blocks, self.parent.ff):
            old = block.ff
            ff = nn.Sequential(nn.Linear(D, width), nn.GELU(), nn.Linear(width, D))
            with torch.no_grad():
                ff[0].weight.copy_(old[0].weight[:width])
                ff[0].bias.copy_(old[0].bias[:width])
                ff[2].weight.copy_(old[2].weight[:, :width])
                ff[2].bias.copy_(old[2].bias)
            block.ff = ff
        self.arm = arm

    def measure_routes(self, enabled=True):
        for side in self.parent.sides:
            side.measure = enabled

    def routing_metrics(self):
        return [dict(side.stats) for side in self.parent.sides]

    def routing_cost(self):
        """Mean probability of any optional activation, a differentiable proxy."""
        scores = [side.soft_cost for side in self.parent.sides if side.soft_cost is not None]
        return torch.stack(scores).mean() if scores else self.parent.core.emb.weight.new_zeros(())


def build_cog(arm, seed):
    parent = {"A": "NN_native_native", "B": "BB_native_native", "C": "SS_native_native"}
    if arm in parent:
        return WholeCOG(parent[arm], seed)
    if arm in ("D", "E", "F", "G", "H"):
        return AdaptiveCOG(arm, seed)
    raise ValueError(f"Unknown COG-BALANCE arm {arm!r}")
