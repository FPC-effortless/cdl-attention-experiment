"""Phase-1 CASM-S router and topological Boolean executor.

The router receives node-local structural metadata only. It never receives the
episode's true wiring or runtime Boolean values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .generator import Episode
from .grammar import Edge, Op

OP_IDS = {Op.INPUT: 0, Op.AND: 1, Op.OR: 2, Op.XOR: 3, Op.NOT: 4}
REL_IDS = {"arg1-of": 0, "arg2-of": 1}
STRUCTURE_DIM = 7
INITIAL_LOGIT_GAIN = 7.5

@dataclass
class ForwardResult:
    output: Tensor
    gates: Tensor
    alpha: Tensor
    node_values: Tensor

def structural_tensor(episode: Episode) -> Tensor:
    """Encode node type/position only; true wiring is deliberately excluded."""
    n = len(episode.nodes)
    denom = max(1, n - 1)
    rows = []
    for node in episode.nodes:
        one_hot = [0.0] * len(OP_IDS)
        one_hot[OP_IDS[node.op]] = 1.0
        rows.append(one_hot + [node.index / denom, float(node.arity) / 2.0])
    return torch.tensor(rows, dtype=torch.float32)

class FactorizedRouter(nn.Module):
    """CASM-S router with independently encoded source/target nodes."""
    def __init__(self, d_model: int = 32, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.logit_gain = INITIAL_LOGIT_GAIN
        self.encoder = nn.Sequential(nn.Linear(STRUCTURE_DIM, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.q.weight, mean=0.0, std=0.40)
        nn.init.normal_(self.k.weight, mean=0.0, std=0.40)

    def logits(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        h = self.encoder(structure)
        q, k = self.q(h), self.k(h)
        scale = math.sqrt(q.shape[-1])
        return torch.stack([
            self.logit_gain * (q[e.dst] * k[e.src]).sum() / scale
            + self.relation_bias[e.port]
            for e in edges
        ])

    def gates(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        return torch.sigmoid(self.logits(structure, edges) / self.temperature)

class StaticMask(nn.Module):
    """Static physical-edge control over the fixed N=4 substrate."""
    def __init__(self, max_nodes: int = 4) -> None:
        super().__init__()
        if max_nodes != 4:
            raise ValueError("Phase-1 first run fixes the physical substrate at N=4")
        self.edge_keys = tuple((src, dst, port) for dst in range(2, max_nodes) for port in range(2) for src in range(dst))
        self.logits = nn.Parameter(torch.zeros(len(self.edge_keys)))
        self._index = {key: i for i, key in enumerate(self.edge_keys)}

    def gates(self, _: Tensor, edges: Iterable[Edge]) -> Tensor:
        return torch.sigmoid(torch.stack([self.logits[self._index[(e.src, e.dst, e.port)]] for e in edges]))

class CASMExecutor(nn.Module):
    """Differentiable single-pass topological Boolean execution."""
    def __init__(self, num_relations: int = 2, c: float = 1.0) -> None:
        super().__init__()
        if num_relations != 2:
            raise ValueError("Phase 1 has exactly arg1-of and arg2-of relations")
        self.eta = nn.Parameter(torch.full((num_relations,), math.log(math.e - 1.0)))
        self.register_buffer("c", torch.tensor(float(c)))

    def alpha(self) -> Tensor:
        return self.c * F.softplus(self.eta)

    def execute_assignment(self, episode: Episode, gates: Tensor, assignment: Tensor) -> Tensor:
        if gates.shape != (len(episode.candidate_edges),):
            raise ValueError("gate shape must equal candidate edge count")
        values = [torch.zeros((), device=assignment.device) for _ in episode.nodes]
        for i in range(len(episode.inputs)):
            values[i] = assignment[i]
        alpha = self.alpha()
        for node in episode.nodes:
            if node.op is Op.INPUT:
                continue
            port_values = []
            for port in range(node.arity):
                indices = [k for k, e in enumerate(episode.candidate_edges) if e.dst == node.index and e.port == port]
                terms = [gates[k] * alpha[episode.candidate_edges[k].port] * values[episode.candidate_edges[k].src] for k in indices]
                port_values.append(torch.stack(terms).sum() if terms else torch.zeros((), device=assignment.device))
            if node.op is Op.AND:
                value = port_values[0] * port_values[1]
            elif node.op is Op.OR:
                value = port_values[0] + port_values[1] - port_values[0] * port_values[1]
            elif node.op is Op.XOR:
                value = port_values[0] + port_values[1] - 2 * port_values[0] * port_values[1]
            elif node.op is Op.NOT:
                value = 1.0 - port_values[0]
            else:
                raise ValueError(f"Unsupported op: {node.op}")
            values[node.index] = value
        return values[episode.output]

def batch_forward(episodes: list[Episode], router: nn.Module, executor: CASMExecutor):
    losses, gates_all, outputs = [], [], []
    for episode in episodes:
        gates = router.gates(structural_tensor(episode), episode.candidate_edges)
        for row, target in enumerate(episode.truth_table):
            bits = [(row >> (len(episode.inputs) - 1 - i)) & 1 for i in range(len(episode.inputs))]
            assignment = torch.tensor(bits, dtype=torch.float32, device=gates.device)
            pred = executor.execute_assignment(episode, gates, assignment)
            losses.append((pred - float(target)) ** 2)
            outputs.append(pred)
        gates_all.append(gates)
    return torch.stack(losses).mean(), torch.cat(gates_all), torch.stack(outputs)

def gate_diagnostics(router: nn.Module, episodes: list[Episode]) -> dict[str, float]:
    with torch.no_grad():
        gates, logits = [], []
        for e in episodes:
            s = structural_tensor(e)
            if isinstance(router, FactorizedRouter):
                l = router.logits(s, e.candidate_edges)
                g = torch.sigmoid(l / router.temperature)
            else:
                g = router.gates(s, e.candidate_edges)
                l = torch.logit(g.clamp(1e-6, 1 - 1e-6))
            gates.append(g); logits.append(l)
        g, l = torch.cat(gates), torch.cat(logits)
        return {"gate_mean": float(g.mean()), "gate_std": float(g.std(unbiased=False)), "logit_std": float(l.std(unbiased=False)), "min_gate": float(g.min()), "max_gate": float(g.max())}

def gate_0(router: nn.Module, episodes: list[Episode]) -> dict[str, float]:
    d = gate_diagnostics(router, episodes)
    if not 0.45 <= d["gate_mean"] <= 0.55:
        raise AssertionError(f"Gate 0 mean(g) failed: {d}")
    if not 0.05 <= d["logit_std"] <= 0.40:
        raise AssertionError(f"Gate 0 logit std failed: {d}")
    if not d["min_gate"] > 0.10 or not d["max_gate"] < 0.90:
        raise AssertionError(f"Gate 0 saturation failed: {d}")
    return d

def assert_integrity(router: nn.Module, executor: CASMExecutor, episodes: list[Episode]) -> None:
    """Gate 6: relation-indexed alpha and structure/value separation."""
    assert executor.eta.shape == (2,)
    assert torch.all(executor.alpha() >= 0)
    assert not any(name == "alpha_edge" for name, _ in executor.named_parameters())
    assert getattr(router, "relation_bias", torch.zeros(2)).shape == (2,)
    for e in episodes:
        for edge in e.candidate_edges:
            assert edge.port in REL_IDS.values()
        assert structural_tensor(e).shape == (len(e.nodes), STRUCTURE_DIM)
