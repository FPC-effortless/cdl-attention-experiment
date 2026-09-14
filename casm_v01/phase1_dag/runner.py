"""Phase-1 runner for benchmark falsification and conditional routing.

The first executable claim is deliberately narrow:
1. all-candidate copy-mask must fail on an ambiguous substrate;
2. a fixed static mask is the non-conditional control;
3. a small op x argument-role conditional router is the first CASM test.

Compositional OOD is NOT claimed here: the four-op grammar does not provide a
fair held-out-combination test for a factorized router. That experiment is a
separate follow-up after the benchmark passes these controls.
"""
from __future__ import annotations

import argparse
import json
import random

import torch

from .diagnostics import (
    _execute_with_gates,
    gate0_viability,
    gate1_causality,
    gate2_router_gradient,
    gate3_structural_sensitivity,
    gate4_counterfactual,
    gate6_integrity,
)
from .generator import BooleanDAGGenerator
from .model import CASMS, StaticMask, copy_mask_gates
from .pairs import rewire_episode

SEED = 20260914


def edge_metrics(episodes, gates):
    tp = fp = fn = total = 0
    chunks = []
    for b, ep in enumerate(episodes):
        n = len(ep.candidate_edges)
        gb = gates[b, :n]
        chunks.append(gb)
        truth = ep.true_edge_set
        for k, e in enumerate(ep.candidate_edges):
            pred = bool(float(gb[k]) >= 0.5)
            real = (e.src, e.dst, e.port) in truth
            tp += int(pred and real)
            fp += int(pred and not real)
            fn += int((not pred) and real)
            total += 1
    all_g = torch.cat(chunks) if chunks else torch.empty(0)
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "mean_gate": float(all_g.mean()) if all_g.numel() else 0.0,
        "candidate_edges": total,
    }


def expected_edge_ratio(episodes):
    candidates = sum(len(ep.candidate_edges) for ep in episodes)
    true = sum(len(ep.true_edges) for ep in episodes)
    return true / max(1, candidates)


def run_copy_mask_preflight(model, episodes):
    """Falsify the substrate before any learned router is trained."""
    device = next(model.parameters()).device
    rows = []
    correct = 0
    for ep in episodes:
        x = torch.tensor(ep.input_values, dtype=torch.float32, device=device)
        g = copy_mask_gates([ep], device=device)[0]
        y = _execute_with_gates(model, ep, x, g)
        correct += int((float(y) >= 0.5) == bool(ep.target))
        rows.append(g)
    padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    return {
        "task_accuracy": correct / max(1, len(episodes)),
        **edge_metrics(episodes, padded),
        "uses_hidden_oracle": False,
        "purpose": "benchmark_falsification",
    }


def make_batch(episodes, device):
    n = max(len(e.inputs) for e in episodes)
    return torch.tensor(
        [list(e.input_values) + [0] * (n - len(e.inputs)) for e in episodes],
        dtype=torch.float32,
        device=device,
    )


def train_model(model, train, test, epochs=50, warmup=5, budget_target=None, lr=2e-3, budget_weight=0.05):
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    history = []
    budget_target = expected_edge_ratio(train) if budget_target is None else budget_target
    budget_enabled = False

    for epoch in range(epochs):
        model.train()
        y, g, _ = model(train, make_batch(train, device))
        target = torch.tensor([e.target for e in train], dtype=torch.float32, device=device)
        task = torch.nn.functional.mse_loss(y, target)
        valid = torch.cat([g[b, : len(ep.candidate_edges)] for b, ep in enumerate(train)])
        budget = (valid.mean() - budget_target).pow(2)

        if epoch >= warmup and not budget_enabled:
            pair = (train[0], rewire_episode(train[0], SEED + epoch))
            budget_enabled = bool(gate3_structural_sensitivity(model, *pair).get("pass", False))

        loss = task + budget_weight * budget if budget_enabled else task
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append({
            "epoch": epoch + 1,
            "task_loss": float(task.detach()),
            "budget_loss": float(budget.detach()),
            "budget_target": budget_target,
            "budget_enabled": budget_enabled,
        })

    model.eval()
    with torch.no_grad():
        pred, _, _ = model(test, make_batch(test, device))
    truth = torch.tensor([e.target for e in test], device=device).bool()
    return {
        "test_accuracy": float(((pred >= 0.5) == truth).float().mean()),
        "history": history,
        "budget_target": budget_target,
    }


def _find_passing_gate(model, episodes, gate_fn):
    """Run a causal gate over several episodes; do not let one dormant input fail it."""
    failures = []
    for ep in episodes:
        result = gate_fn(model, ep)
        if result.get("pass"):
            result["episode_index"] = episodes.index(ep)
            return result
        failures.append(result)
    result = failures[0] if failures else {"pass": False}
    result["episodes_checked"] = len(episodes)
    result["reason"] = result.get("reason", "no episode produced a causal witness")
    return result


def run(seed=SEED, train_size=128, test_size=64):
    random.seed(seed)
    torch.manual_seed(seed)
    gen = BooleanDAGGenerator(max_nodes=10, min_nodes=4, seed=seed)
    train = gen.sample_batch(train_size)
    test = gen.sample_batch(test_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    casm = CASMS(10, 32, 2.0, seed).to(device)
    static = StaticMask(10, 32, 2.0, seed).to(device)

    copy = run_copy_mask_preflight(casm, test[:32])
    if copy["task_accuracy"] > 0.95:
        raise RuntimeError("COPY_MASK_PREFLIGHT_FAILED: candidate substrate is too easy; do not interpret learned routing until the generator is fixed.")

    paired = (train[0], rewire_episode(train[0], seed + 1))
    results = {
        "seed": seed,
        "execution": "single-pass topological DAG",
        "copy_mask_preflight": copy,
        "expected_train_edge_ratio": expected_edge_ratio(train),
        "calibration": {"casm": gate0_viability(casm, test[:16]), "static": gate0_viability(static, test[:16])},
        "claims": {"conditional_routing": "in-scope", "compositional_ood": "deferred", "reason": "four-op grammar does not support a fair held-out role-combination test"},
    }

    results["static_train"] = train_model(static, train, test)
    results["casm_train"] = train_model(casm, train, test)

    x = make_batch(test, device)
    with torch.no_grad():
        _, gc, _ = casm(test, x)
        _, gs, _ = static(test, x)
    results["routing"] = {"casm": edge_metrics(test, gc), "static": edge_metrics(test, gs)}

    # Causal diagnostics must search for an executable witness. A single
    # sampled Boolean assignment can make a real edge locally dormant; failing
    # on that sample would reproduce the diagnostic bug we are trying to avoid.
    results["gate1_causality"] = _find_passing_gate(casm, test, gate1_causality)
    results["gate2_router_gradient"] = gate2_router_gradient(casm, train[:16])
    results["gate3_structural_sensitivity"] = gate3_structural_sensitivity(casm, *paired)
    results["gate4_counterfactual"] = _find_passing_gate(casm, test, gate4_counterfactual)
    results["gate6_integrity"] = gate6_integrity(casm, test[:8])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--output", default="casm_v01/phase1_dag/results.json")
    args = p.parse_args()
    result = run(args.seed, args.train_size, args.test_size)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
