"""Phase-1 CASM-S router and topological Boolean executor.

Control-flow structure and runtime values are separate: the router receives only
program structure; runtime Boolean values enter only the execution loop.
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


@dataclass
class ForwardResult:
    output: Tensor
    gates: Tensor
    alpha: Tensor
    node_values: Tensor


def _node_depths(episode: Episode) -> list[int]:
    incoming = {n.index: [] for n in episode.nodes}
    for e in episode.true_edges:
        incoming[e.dst].append(e.src)
    depth = [0] * len(episode.nodes)
    for n in episode.nodes:
        if incoming[n.index]:
            depth[n.index] = 1 + max(depth[s] for s in incoming[n.index])
    return depth


def structural_tensor(episode: Episode) -> Tensor:
    """Encode program structure without exposing runtime values.

    Source indices are structural metadata, not a runtime adjacency mask.  This
    lets the router parse the program while keeping the copy-mask control blind
    to the episode wiring.
    """
    n = len(episode.nodes)
    depths = _node_depths(episode)
    sources = {i: [-1, -1] for i in range(n)}
    for e in episode.true_edges:
        sources[e.dst][e.port] = e.src
    denom = max(1, n - 1)
    rows = []
    for node in episode.nodes:
        s0, s1 = sources[node.index]
        rows.append([
            float(OP_IDS[node.op]),
            node.index / denom,
            depths[node.index] / denom,
            float(node.arity),
            (s0 + 1) / (n + 1),
            (s1 + 1) / (n + 1),
        ])
    return torch.tensor(rows, dtype=torch.float32)


class FactorizedRouter(nn.Module):
    """CASM-S factorized source/target router.

    No parameter is indexed by a physical edge or an op-pair.  Source and target
    representations are formed independently and combined by bilinear scoring.
    """

    def __init__(self, d_model: int = 32, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.Linear(6, d_model), nn.Tanh(), nn.Linear(d_model, d_model)
        )
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(2))
        self._init_router()

    def _init_router(self) -> None:
        nn.init.normal_(self.q.weight, mean=0.0, std=0.10)
        nn.init.normal_(self.k.weight, mean=0.0, std=0.10)

    def logits(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        h = self.encoder(structure)
        q = self.q(h)
        k = self.k(h)
        scale = math.sqrt(q.shape[-1])
        return torch.stack([
            (q[e.dst] * k[e.src]).sum() / scale + self.relation_bias[e.port]
            for e in edges
        ])

    def gates(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        return torch.sigmoid(self.logits(structure, edges) / self.temperature)


class StaticMask(nn.Module):
    """Static physical-edge control over the fixed N=4 Phase-1 substrate."""

    def __init__(self, max_nodes: int = 4) -> None:
        super().__init__()
        if max_nodes != 4:
            raise ValueError("Phase-1 first run fixes the physical substrate at N=4")
        self.edge_keys = tuple(
            (src, dst, port)
            for dst in range(2, max_nodes)
            for port in range(2)
            for src in range(dst)
        )
        self.logits = nn.Parameter(torch.zeros(len(self.edge_keys)))
        self._index = {key: i for i, key in enumerate(self.edge_keys)}

    def gates(self, _: Tensor, edges: Iterable[Edge]) -> Tensor:
        return torch.sigmoid(torch.stack([
            self.logits[self._index[(e.src, e.dst, e.port)]] for e in edges
        ]))


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
        values = torch.zeros(len(episode.nodes), device=assignment.device)
        values[: len(episode.inputs)] = assignment
        alpha = self.alpha()
        for node in episode.nodes:
            if node.op is Op.INPUT:
                continue
            port_values = []
            for port in range(node.arity):
                indices = [k for k, e in enumerate(episode.candidate_edges)
                           if e.dst == node.index and e.port == port]
                s = torch.zeros((), device=assignment.device)
                for k in indices:
                    e = episode.candidate_edges[k]
                    s = s + gates[k] * alpha[e.port] * values[e.src]
                port_values.append(s)
            if node.op is Op.AND:
                values[node.index] = port_values[0] * port_values[1]
            elif node.op is Op.OR:
                values[node.index] = port_values[0] + port_values[1] - port_values[0] * port_values[1]
            elif node.op is Op.XOR:
                values[node.index] = port_values[0] + port_values[1] - 2 * port_values[0] * port_values[1]
            elif node.op is Op.NOT:
                values[node.index] = 1.0 - port_values[0]
        return values[episode.output]


def batch_forward(episodes: list[Episode], router: nn.Module, executor: CASMExecutor):
    losses, gates_all, outputs = [], [], []
    for episode in episodes:
        structure = structural_tensor(episode)
        gates = router.gates(structure, episode.candidate_edges)
        for row, target in enumerate(episode.truth_table):
            bits = [(row >> (len(episode.inputs) - 1 - i)) & 1
                    for i in range(len(episode.inputs))]
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
            gates.append(g)
            logits.append(l)
        g, l = torch.cat(gates), torch.cat(logits)
        return {
            "gate_mean": float(g.mean()),
            "gate_std": float(g.std(unbiased=False)),
            "logit_std": float(l.std(unbiased=False)),
            "min_gate": float(g.min()),
            "max_gate": float(g.max()),
        }


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
    """Gate 6: relation-indexed alpha only; no physical-edge alpha table."""
    assert executor.eta.shape == (2,)
    assert torch.all(executor.alpha() >= 0)
    assert not any(name == "alpha_edge" for name, _ in executor.named_parameters())
    assert getattr(router, "relation_bias", torch.zeros(2)).shape == (2,)
    for e in episodes:
        for edge in e.candidate_edges:
            assert edge.port in REL_IDS.values()
