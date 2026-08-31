from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import torch

from casm.model import CASM, CASMConfig

PAD = 256
CONTEXT = 0
QUERY = 1
RECALL = 2
FILL = 3
DATA_MIN = 8
DATA_MAX = 63
TASKS = ("single_key", "multi_key", "delayed_query", "noisy_key", "multi_hop")


def tiny_cfg(variant: str) -> CASMConfig:
    cfg = CASMConfig(
        vocab_size=260,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        d_ff=128,
        chunk_size=24,
        memory_slots=8,
        state_slots=2,
        memory_dim=24,
        dropout=0.0,
        compression_future_tokens=8,
        mtp_horizons=1,
        mtp_loss_weight=0.0,
        verifier_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    if variant == "local_only":
        return replace(cfg, use_memory=False, use_compression_score=False, compression_loss_weight=0.0)
    if variant == "qk_memory":
        return replace(cfg, use_memory=True, use_compression_score=False, compression_loss_weight=0.0)
    if variant == "compression_runtime":
        return replace(cfg, use_memory=True, use_compression_score=True, compression_loss_weight=0.15)
    raise ValueError(f"unknown variant: {variant}")


def _choice(rng: random.Random, exclude: Iterable[int] = ()) -> int:
    excluded = set(exclude)
    while True:
        x = rng.randint(DATA_MIN, DATA_MAX)
        if x not in excluded:
            return x


def _noise_fill(rng: random.Random, n: int, exclude: Iterable[int] = ()) -> list[int]:
    return [_choice(rng, exclude) for _ in range(n)]


def make_example(task: str, rng: random.Random) -> tuple[list[int], list[int], int]:
    """Return (24-token context, 7-token query prefix, target token).

    Context and query are deliberately split at CASM's chunk boundary. Local
    attention therefore cannot directly see context tokens while answering the
    query; only carried episodic/persistent state can cross the boundary.
    """
    if task not in TASKS:
        raise ValueError(task)

    if task == "single_key":
        key = _choice(rng)
        value = _choice(rng, [key])
        context = [CONTEXT, key, value]
        context += _noise_fill(rng, 21, [key, value])
        query_key, target = key, value

    elif task == "multi_key":
        used: list[int] = []
        pairs: list[tuple[int, int]] = []
        for _ in range(5):
            k = _choice(rng, used)
            used.append(k)
            v = _choice(rng, used)
            used.append(v)
            pairs.append((k, v))
        context = []
        for k, v in pairs:
            context.extend([CONTEXT, k, v])
        context += _noise_fill(rng, 24 - len(context), used)
        query_key, target = pairs[rng.randrange(len(pairs))]

    elif task == "delayed_query":
        key = _choice(rng)
        value = _choice(rng, [key])
        # Put the target binding at the beginning and consume the remaining
        # context with unrelated bindings/filler before the delayed query.
        context = [CONTEXT, key, value]
        used = [key, value]
        while len(context) <= 18:
            k = _choice(rng, used)
            used.append(k)
            v = _choice(rng, used)
            used.append(v)
            context.extend([CONTEXT, k, v])
        context = (context + _noise_fill(rng, 24, used))[:24]
        query_key, target = key, value

    elif task == "noisy_key":
        key = _choice(rng)
        value = _choice(rng, [key])
        context = [CONTEXT, key, value]
        used = [key, value]
        # Several decoy key/value bindings plus unrelated noise.
        for _ in range(5):
            dk = _choice(rng, used)
            used.append(dk)
            dv = _choice(rng, used)
            used.append(dv)
            context.extend([CONTEXT, dk, dv])
        context = (context + _noise_fill(rng, 24, used))[:24]
        query_key, target = key, value

    else:  # multi_hop
        start = _choice(rng)
        mid = _choice(rng, [start])
        target = _choice(rng, [start, mid])
        # Two explicit edges are required: start -> mid -> target.
        context = [CONTEXT, start, mid, CONTEXT, mid, target]
        used = [start, mid, target]
        for _ in range(4):
            dk = _choice(rng, used)
            used.append(dk)
            dv = _choice(rng, used)
            used.append(dv)
            context.extend([CONTEXT, dk, dv])
        context = (context + _noise_fill(rng, 24, used))[:24]
        query_key = start

    assert len(context) == 24
    query_prefix = [QUERY, query_key, RECALL, FILL, FILL, FILL, FILL]
    return context, query_prefix, target


def make_batch(task: str, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    contexts, queries, targets = [], [], []
    for _ in range(batch_size):
        context, query, target = make_example(task, rng)
        contexts.append(context)
        queries.append(query)
        targets.append(target)
    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(queries, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def _carried_tokens(contexts: torch.Tensor, queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.cat([contexts, queries, targets[:, None]], dim=1)


def _query_only_tokens(queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.cat([queries, targets[:, None]], dim=1)


def target_only_weights(tokens: torch.Tensor) -> torch.Tensor:
    # CASM predicts tokens[:, 1:] from tokens[:, :-1]. The final prediction is
    # the recall target; train only that position so prompt-template modeling
    # cannot dominate the benchmark objective.
    w = torch.zeros(tokens.shape[0], tokens.shape[1] - 1, dtype=torch.float32, device=tokens.device)
    w[:, -1] = 1.0
    return w


def last_token_logits(model: CASM, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens, return_aux=False)["logits"][:, -1]


@torch.inference_mode()
def evaluate_state_controls(model: CASM, task: str, *, seed: int, batches: int, batch_size: int) -> dict[str, float]:
    model.eval()
    carried_correct = reset_correct = shuffled_correct = total = 0
    for i in range(batches):
        contexts, queries, targets = make_batch(task, batch_size, seed + i * 7919)
        carried = _carried_tokens(contexts, queries, targets)
        reset = _query_only_tokens(queries, targets)
        shuffled_contexts = torch.roll(contexts, shifts=1, dims=0)
        shuffled = _carried_tokens(shuffled_contexts, queries, targets)
        carried_correct += int((last_token_logits(model, carried).argmax(-1) == targets).sum())
        reset_correct += int((last_token_logits(model, reset).argmax(-1) == targets).sum())
        shuffled_correct += int((last_token_logits(model, shuffled).argmax(-1) == targets).sum())
        total += targets.numel()
    carried_acc = carried_correct / max(1, total)
    reset_acc = reset_correct / max(1, total)
    shuffled_acc = shuffled_correct / max(1, total)
    return {
        "carried_accuracy": carried_acc,
        "reset_accuracy": reset_acc,
        "shuffled_accuracy": shuffled_acc,
        "carry_reset_delta": carried_acc - reset_acc,
        "carry_shuffled_delta": carried_acc - shuffled_acc,
        "n": total,
    }


def train_variant(variant: str, task: str, *, seed: int, steps: int, batch_size: int, lr: float) -> tuple[CASM, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = tiny_cfg(variant)
    model = CASM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        contexts, queries, targets = make_batch(task, batch_size, seed + step * 104729)
        tokens = _carried_tokens(contexts, queries, targets)
        out = model(tokens, return_aux=True, target_weights=target_only_weights(tokens))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, time.perf_counter() - started


def run_matrix(*, variants: list[str], tasks: list[str], seeds: list[int], steps: int, batch_size: int, eval_batches: int, lr: float) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        for task in tasks:
            for seed in seeds:
                model, elapsed = train_variant(variant, task, seed=seed, steps=steps, batch_size=batch_size, lr=lr)
                metrics = evaluate_state_controls(
                    model,
                    task,
                    seed=seed + 10_000_000,
                    batches=eval_batches,
                    batch_size=batch_size,
                )
                rows.append(
                    {
                        "variant": variant,
                        "task": task,
                        "seed": seed,
                        "parameters": model.parameter_count(),
                        "train_seconds": elapsed,
                        **metrics,
                    }
                )
    summary: dict[str, dict[str, float]] = {}
    for variant in variants:
        vr = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "mean_carried_accuracy": sum(float(r["carried_accuracy"]) for r in vr) / len(vr),
            "mean_reset_accuracy": sum(float(r["reset_accuracy"]) for r in vr) / len(vr),
            "mean_shuffled_accuracy": sum(float(r["shuffled_accuracy"]) for r in vr) / len(vr),
            "mean_carry_reset_delta": sum(float(r["carry_reset_delta"]) for r in vr) / len(vr),
            "mean_carry_shuffled_delta": sum(float(r["carry_shuffled_delta"]) for r in vr) / len(vr),
            "mean_train_seconds": sum(float(r["train_seconds"]) for r in vr) / len(vr),
        }
    return {"contract": "casm-a4-state-preserved-v1", "rows": rows, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="local_only,qk_memory,compression_runtime")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", type=Path, default=Path("benchmark-output/state-preserved.json"))
    args = p.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    result = run_matrix(
        variants=[x.strip() for x in args.variants.split(",") if x.strip()],
        tasks=[x.strip() for x in args.tasks.split(",") if x.strip()],
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        lr=args.lr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
