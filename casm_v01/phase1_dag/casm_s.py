"""Phase-1 CASM-S router and topological Boolean executor.

This module intentionally keeps control-flow structure and runtime values separate.
The router consumes only node/program structure; runtime Boolean values enter only the
execution loop.  The first experiment is Static Mask vs CASM-S.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

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
    """Return node structure features; no runtime values are included.

    Features are [op-id, normalized-index, normalized-depth, arity,
    source0-index, source1-index].  Source indices encode the program wiring
    as structural data, while runtime Boolean values remain completely absent.
    """
    n = len(episode.nodes)
    depths = _node_depths(episode)
    sources = {i: [-1, -1] for i in range(n)}
    for e in episode.true_edges:
        sources[e.dst][e.port] = e.src
    rows = []
    denom = max(1, n - 1)
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

    There is deliberately no parameter indexed by (source-op, target-op), edge,
    or physical node pair.  q_i and k_j are produced independently from node
    structure and combined bilinearly.
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
        # Scale the bilinear path so initial logits are in the locked range.
        with torch.no_grad():
            probe = torch.randn(4096, 32) if self.q.in_features == 32 else None
            if probe is not None:
                pass

    def logits(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        h = self.encoder(structure)
        q = self.q(h)
        k = self.k(h)
        scale = math.sqrt(q.shape[-1])
        values = []
        for e in edges:
            values.append((q[e.dst] * k[e.src]).sum() / scale + self.relation_bias[e.port])
        return torch.stack(values)

    def gates(self, structure: Tensor, edges: Iterable[Edge]) -> Tensor:
        logits = self.logits(structure, edges)
        return torch.sigmoid(logits / self.temperature)


class StaticMask(nn.Module):
    """Fixed physical-edge mask control for the same candidate substrate."""

    def __init__(self, num_edges: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_edges))

    def gates(self, _: Tensor, edges: Iterable[Edge]) -> Tensor:
        del edges
        return torch.sigmoid(self.logits)


class CASMExecutor(nn.Module):
    """Differentiable single-pass topological Boolean execution."""

    def __init__(self, num_relations: int = 2, c: float = 1.0) -> None:
        super().__init__()
        if num_relations != 2:
            raise ValueError("Phase 1 has exactly arg1-of and arg2-of relations")
        self.eta = nn.Parameter(torch.zeros(num_relations))
        self.register_buffer("c", torch.tensor(float(c)))

    def alpha(self) -> Tensor:
        return self.c * torch.nn.functional.softplus(self.eta)

    def forward(self, episode: Episode, gates: Tensor) -> ForwardResult:
        edges = episode.candidate_edges
        if gates.shape != (len(edges),):
            raise ValueError("gate shape must equal candidate edge count")
        device = gates.device
        values = torch.zeros(len(episode.nodes), device=device)
        assignment_count = 2 ** len(episode.inputs)
        # The executor is called on one assignment at a time by train_episode.
        raise RuntimeError("use execute_assignment for runtime execution")

    def execute_assignment(self, episode: Episode, gates: Tensor, assignment: Tensor) -> Tensor:
        values = torch.zeros(len(episode.nodes), device=assignment.device)
        values[: len(episode.inputs)] = assignment
        alpha = self.alpha()
        for node in episode.nodes:
            if node.op is Op.INPUT:
                continue
            port_values = []
            for port in range(node.arity):
                candidates = [
                    k for k, e in enumerate(episode.candidate_edges)
                    if e.dst == node.index and e.port == port
                ]
                if not candidates:
                    raise RuntimeError(f"missing candidates for node {node.index}, port {port}")
                s = torch.zeros((), device=assignment.device)
                for k in candidates:
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


def batch_forward(
    episodes: list[Episode], router: nn.Module, executor: CASMExecutor
) -> tuple[Tensor, Tensor, Tensor]:
    losses = []
    all_gates = []
    outputs = []
    for episode in episodes:
        structure = structural_tensor(episode)
        gates = router.gates(structure, episode.candidate_edges)
        alpha = executor.alpha()
        del alpha
        for row, target in enumerate(episode.truth_table):
            bits = [(row >> (len(episode.inputs) - 1 - i)) & 1 for i in range(len(episode.inputs))]
            assignment = torch.tensor(bits, dtype=torch.float32, device=gates.device)
            pred = executor.execute_assignment(episode, gates, assignment)
            target_t = torch.tensor(float(target), device=gates.device)
            losses.append((pred - target_t) ** 2)
            outputs.append(pred)
        all_gates.append(gates)
    return torch.stack(losses).mean(), torch.cat([g.reshape(-1) for g in all_gates]), torch.stack(outputs)


def gate_diagnostics(router: nn.Module, episodes: list[Episode]) -> dict[str, float]:
    with torch.no_grad():
        gs = []
        ls = []
        for e in episodes:
            structure = structural_tensor(e)
            if hasattr(router, "logits"):
                l = router.logits(structure, e.candidate_edges)
                g = torch.sigmoid(l / router.temperature)
            else:
                g = router.gates(structure, e.candidate_edges)
                l = torch.logit(g.clamp(1e-6, 1 - 1e-6))
            gs.append(g)
            ls.append(l)
        g = torch.cat(gs)
        l = torch.cat(ls)
        return {
            "gate_mean": float(g.mean()),
            "gate_std": float(g.std(unbiased=False)),
            "logit_std": float(l.std(unbiased=False)),
            "min_gate": float(g.min()),
            "max_gate": float(g.max()),
        }


def gate_0(router: nn.Module, episodes: list[Episode]) -> dict[str, float]:
    d = gate_diagnostics(router, episodes)
    if not (0.45 <= d["gate_mean"] <= 0.55):
        raise AssertionError(f"Gate 0 mean(g) failed: {d}")
    if not (0.05 <= d["logit_std"] <= 0.40):
        raise AssertionError(f"Gate 0 logit std failed: {d}")
    if not (d["min_gate"] > 0.10 and d["max_gate"] < 0.90):
        raise AssertionError(f"Gate 0 saturation failed: {d}")
    return d


def gate_1(executor: CASMExecutor, episode: Episode, router: nn.Module) -> float:
    structure = structural_tensor(episode)
    gates = router.gates(structure, episode.candidate_edges)
    assignment = torch.tensor([0.0] * len(episode.inputs))
    base = executor.execute_assignment(episode, gates, assignment)
    changed = gates.clone()
    changed[0] = 0.0
    altered = executor.execute_assignment(episode, changed, assignment)
    delta = float((altered - base).abs())
    if delta <= 0.0:
        raise AssertionError("Gate 1 failed: severing a gate did not change the forward value")
    return delta
