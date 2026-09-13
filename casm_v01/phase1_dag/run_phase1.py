"""Run the minimal Phase-1 Static Mask vs CASM-S experiment.

This is deliberately small and CPU-friendly. It performs the mechanical gates first,
then a task-only warm-up, then introduces the batch-level activation budget.
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from torch import nn

from .casm_s import CASMExecutor, FactorizedRouter, StaticMask, batch_forward, gate_0, assert_integrity, structural_tensor
from .generator import Episode, generate_episode
from .oracle import evaluate_episode


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def episodes(count: int, offset: int = 0) -> list[Episode]:
    return [generate_episode(seed=offset + s, n_inputs=2, n_ops=2) for s in range(count)]


def budget_loss(gates: torch.Tensor, target: float) -> torch.Tensor:
    return (gates.mean() - target).pow(2)


def train_model(model: nn.Module, ex: CASMExecutor, train_eps: list[Episode], steps: int,
                warmup: int, rho_star: float, lr: float, budget_weight: float) -> dict:
    params = list(model.parameters()) + list(ex.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        task, gates, _ = batch_forward(train_eps, model, ex)
        loss = task
        if step >= warmup:
            loss = loss + budget_weight * budget_loss(gates, rho_star)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": step + 1, "task_loss": float(task),
                            "budget_loss": float(budget_loss(gates, rho_star)),
                            "gate_mean": float(gates.mean())})
    return {"history": history}


def exact_accuracy(model: nn.Module, ex: CASMExecutor, eps: list[Episode]) -> float:
    correct = total = 0
    with torch.no_grad():
        for e in eps:
            g = model.gates(structural_tensor(e), e.candidate_edges)
            for row, target in enumerate(e.truth_table):
                bits = [(row >> (len(e.inputs) - 1 - i)) & 1 for i in range(len(e.inputs))]
                pred = ex.execute_assignment(e, g, torch.tensor(bits, dtype=torch.float32))
                correct += int((pred.round().item()) == target)
                total += 1
    return correct / total


def routing_pr(model: nn.Module, eps: list[Episode], threshold: float = 0.5) -> dict:
    tp = fp = fn = 0
    with torch.no_grad():
        for e in eps:
            g = model.gates(structural_tensor(e), e.candidate_edges)
            true = e.true_edge_set
            for edge, value in zip(e.candidate_edges, g.tolist()):
                hit = (edge.src, edge.dst, edge.port) in true
                pred = value >= threshold
                tp += int(pred and hit)
                fp += int(pred and not hit)
                fn += int((not pred) and hit)
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--train", type=int, default=64)
    args = ap.parse_args()
    seed_all(args.seed)

    train_eps = episodes(args.train, 0)
    test_eps = episodes(32, 1000)
    router = FactorizedRouter()
    ex = CASMExecutor()
    assert_integrity(router, ex, train_eps)
    gate0 = gate_0(router, train_eps)
    before = routing_pr(router, train_eps)

    # Gate 2: autograd path exists before optimization.
    task, _, _ = batch_forward(train_eps[:2], router, ex)
    task.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in router.parameters() if p.grad is not None)
    if not grad_norm > 0:
        raise AssertionError("Gate 2 failed: router gradient is zero")
    router.zero_grad(set_to_none=True)
    ex.zero_grad(set_to_none=True)

    train_model(router, ex, train_eps, args.steps, args.warmup, 0.40, 2e-3, 0.10)
    result = {
        "seed": args.seed,
        "config": {"steps": args.steps, "warmup": args.warmup, "rho_star": 0.40,
                    "temperature": router.temperature, "alpha_init": ex.alpha().detach().tolist()},
        "gate_0": gate0,
        "gate_2_router_grad_sum": grad_norm,
        "routing_before": before,
        "routing_after": routing_pr(router, test_eps),
        "task_accuracy": {"train": exact_accuracy(router, ex, train_eps),
                          "test": exact_accuracy(router, ex, test_eps)},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
