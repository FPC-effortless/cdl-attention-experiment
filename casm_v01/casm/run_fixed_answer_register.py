from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .contextual_data import make_contextual_batch
from .explicit_compute import aggregate_metrics, state_metrics
from .fixed_answer_register import (
    FIXED_REGISTER,
    REGIMES,
    cloned_fixed_answer_models,
    regime_loss,
    sample_random_query_registers,
)
from .weak_supervision import SoftExplicitTransitionModel

HARNESS_VERSION = "casm-x5-fixed-answer-register-v0-2026-08-31"


def train_step(model, optimizer, batch, regime, random_query):
    optimizer.zero_grad(set_to_none=True)
    loss = regime_loss(model, batch, regime, random_query)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}


def answer_hidden_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    hidden = [i for i in range(pred.shape[-1]) if i != FIXED_REGISTER]
    answer_match = pred[:, :, FIXED_REGISTER].eq(target[:, :, FIXED_REGISTER])
    hidden_match = pred[:, :, hidden].eq(target[:, :, hidden])
    return {
        "answer_final_accuracy": answer_match[:, -1].float().mean().item(),
        "answer_step_accuracy": answer_match.float().mean().item(),
        "hidden_register_accuracy": hidden_match.float().mean().item(),
        "hidden_final_exact": hidden_match[:, -1].all(dim=-1).float().mean().item(),
    }


def evaluate(model, *, depth: int, split: str, n: int, batch_size: int, seed: int):
    rows = []
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_contextual_batch(size, depth, seed + offset * 1009, split=split)
        pred = model.rollout_hard(batch)
        row = state_metrics(pred, batch.target_states)
        row.update(answer_hidden_metrics(pred, batch.target_states))
        rows.append(row)
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--eval-seed", type=int, default=20260981)
    p.add_argument("--eval-n", type=int, default=384)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x5-output/results.json"))
    args = p.parse_args()

    if args.train_depth != 8:
        raise ValueError("CASM-X5 v0 preregisters fixed training depth 8")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    seed_model = SoftExplicitTransitionModel(d_model=96)
    models = cloned_fixed_answer_models(seed_model)
    optimizers = {
        regime: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for regime, model in models.items()
    }
    parameters = {regime: model.parameter_count() for regime, model in models.items()}
    assert len(set(parameters.values())) == 1, parameters

    history = []
    random_query_counts = torch.zeros(4, dtype=torch.long)
    for step in range(1, args.steps + 1):
        batch = make_contextual_batch(
            args.batch_size,
            args.train_depth,
            args.seed * 1_000_003 + step * 97,
            split="train",
        )
        random_query = sample_random_query_registers(
            args.batch_size,
            args.seed * 1_000_033 + step * 193,
            device=batch.initial.device,
        )
        random_query_counts += torch.bincount(random_query.cpu(), minlength=4)
        row = {"step": step}
        for regime in REGIMES:
            models[regime].train()
            row[regime] = train_step(
                models[regime], optimizers[regime], batch, regime, random_query
            )
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    suites = [
        ("iid_depth_8", "iid", 8),
        ("composition_depth_12", "composition", 12),
        ("composition_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    evaluation = {}
    for i, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
        for regime in REGIMES:
            evaluation[suite][regime] = evaluate(
                models[regime],
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + i * 100_003,
            )
        print(suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "can permanently hidden working-state registers be recovered from one fixed final answer register?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "train_depth": args.train_depth,
        "fixed_answer_register": FIXED_REGISTER,
        "parameters": parameters,
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "intermediate_state_targets": False,
            "full_final": "all four final register values",
            "random_register": "one uniformly sampled final register value per training example",
            "fixed_register": "final value of register 0 only for every training example",
            "registers_1_2_3_are_never_loss_targets_in_fixed_register": True,
            "query_is_loss_only_not_transition_input": True,
        },
        "random_query_counts": random_query_counts.tolist(),
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
