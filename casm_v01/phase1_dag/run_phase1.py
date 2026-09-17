"""Correct paired Phase-1 Static Mask vs CASM-S experiment runner."""
from __future__ import annotations

import argparse
import json
import random

import torch
from torch import nn

from .benchmark_preflight import run_suite
from .casm_s import CASMExecutor, FactorizedRouter, StaticMask, batch_forward, gate_0, assert_integrity, structural_tensor
from .generator import Episode, generate_episode


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def episodes(count: int, offset: int = 0) -> list[Episode]:
    return [generate_episode(seed=offset + s, n_inputs=2, n_ops=2) for s in range(count)]


def budget_loss(gates: torch.Tensor, target: float) -> torch.Tensor:
    return (gates.mean() - target).pow(2)


def parameter_count(*models: nn.Module) -> int:
    return sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)


def train_model(model: nn.Module, ex: CASMExecutor, train_eps: list[Episode], *, steps: int,
                warmup: int, rho_star: float, lr: float, budget_weight: float) -> dict:
    params = list(model.parameters()) + list(ex.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        task, gates, _ = batch_forward(train_eps, model, ex)
        loss = task + (budget_weight * budget_loss(gates, rho_star) if step >= warmup else 0.0)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    return {"task_loss": float(task), "budget_error": float(budget_loss(gates, rho_star)),
            "gate_mean": float(gates.mean())}


def exact_accuracy(model: nn.Module, ex: CASMExecutor, eps: list[Episode]) -> float:
    correct = total = 0
    with torch.no_grad():
        for e in eps:
            g = model.gates(structural_tensor(e), e.candidate_edges)
            for row, target in enumerate(e.truth_table):
                bits = [(row >> (len(e.inputs) - 1 - i)) & 1 for i in range(len(e.inputs))]
                pred = ex.execute_assignment(e, g, torch.tensor(bits, dtype=torch.float32))
                correct += int(round(float(pred)) == target)
                total += 1
    return correct / max(1, total)


def routing_pr(model: nn.Module, eps: list[Episode], threshold: float = 0.5) -> dict:
    tp = fp = fn = 0
    with torch.no_grad():
        for e in eps:
            g = model.gates(structural_tensor(e), e.candidate_edges)
            true = e.true_edge_set
            for edge, value in zip(e.candidate_edges, g.tolist()):
                hit = (edge.src, edge.dst, edge.port) in true
                pred = value >= threshold
                tp += int(pred and hit); fp += int(pred and not hit); fn += int((not pred) and hit)
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def static_gate0(model: StaticMask, eps: list[Episode]) -> dict:
    with torch.no_grad():
        g = torch.cat([model.gates(structural_tensor(e), e.candidate_edges) for e in eps])
    mean = float(g.mean())
    return {"pass": 0.40 <= mean <= 0.60 and bool(torch.isfinite(g).all()),
            "gate_mean": mean, "gate_std": float(g.std(unbiased=False)),
            "note": "static baseline does not require nonzero logit variance"}


def structural_pair(eps: list[Episode]) -> tuple[Episode, Episode]:
    groups = {}
    for e in eps:
        key = (len(e.nodes), len(e.inputs), len(e.candidate_edges))
        groups.setdefault(key, []).append(e)
    for group in groups.values():
        for a in group:
            sig_a = tuple(n.op.value for n in a.nodes)
            for b in group:
                sig_b = tuple(n.op.value for n in b.nodes)
                if sig_a != sig_b:
                    return a, b
    raise RuntimeError("could not construct a visible structural-sensitivity pair")


def evaluate_model(model: nn.Module, ex: CASMExecutor, train_eps: list[Episode], test_eps: list[Episode]) -> dict:
    return {
        "parameter_count": parameter_count(model, ex),
        "train_accuracy": exact_accuracy(model, ex, train_eps),
        "test_accuracy": exact_accuracy(model, ex, test_eps),
        "train_routing": routing_pr(model, train_eps),
        "test_routing": routing_pr(model, test_eps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--train", type=int, default=64)
    args = ap.parse_args()
    seed_all(args.seed)

    validity = run_suite(seeds=range(8), n_inputs=2, n_ops=2, max_edges=12)
    if not validity["ready_for_router"]:
        raise RuntimeError("Phase-1 benchmark invalid; refusing to train routers")

    train_eps = episodes(args.train, 0)
    test_eps = episodes(32, 1000)
    pair = structural_pair(train_eps)
    models = {"static": (StaticMask(max_nodes=4), CASMExecutor()),
              "casm_s": (FactorizedRouter(), CASMExecutor())}
    result = {"seed": args.seed, "validity": validity, "config": vars(args), "models": {}}

    for name, (model, ex) in models.items():
        assert_integrity(model, ex, train_eps)
        g0 = static_gate0(model, train_eps) if name == "static" else gate_0(model, train_eps)
        task, _, _ = batch_forward(train_eps[:2], model, ex)
        task.backward()
        grad_sum = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        if grad_sum <= 0:
            raise AssertionError(f"{name}: model gradient path is zero")
        model.zero_grad(set_to_none=True); ex.zero_grad(set_to_none=True)
        before = evaluate_model(model, ex, train_eps, test_eps)
        train_stats = train_model(model, ex, train_eps, steps=args.steps, warmup=args.warmup,
                                  rho_star=0.40, lr=2e-3, budget_weight=0.10)
        after = evaluate_model(model, ex, train_eps, test_eps)
        result["models"][name] = {"gate_0": g0, "gradient_sum": grad_sum,
                                   "before": before, "train": train_stats, "after": after}

    router, rex = models["casm_s"]
    with torch.no_grad():
        ga = router.gates(structural_tensor(pair[0]), pair[0].candidate_edges)
        gb = router.gates(structural_tensor(pair[1]), pair[1].candidate_edges)
    diff = float((ga - gb).abs().mean())
    result["gate_3"] = {"pass": diff > 1e-4, "mean_gate_difference": diff,
                         "signature_a": [n.op.value for n in pair[0].nodes],
                         "signature_b": [n.op.value for n in pair[1].nodes]}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
