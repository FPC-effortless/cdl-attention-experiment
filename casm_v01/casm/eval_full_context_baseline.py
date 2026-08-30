from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

from .data import BOS, EOS, PAD, Example
from .full_context_baseline import FullContextTransformer
from .model import CASMConfig
from .process_data import (
    PROCESS_TASKS,
    associative_long,
    corrected_state_long,
    graph_process,
)


def load_model(path: str) -> FullContextTransformer:
    ckpt = torch.load(path, map_location="cpu")
    cfg = CASMConfig(**ckpt["config"])
    model = FullContextTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def prompt_prefix(ex: Example) -> bytes:
    """Return prompt bytes through the final literal ``answer `` marker only."""
    body = ex.text.encode("utf-8", errors="replace")
    marker = b"answer "
    at = body.rfind(marker)
    if at < 0:
        raise ValueError("example has no answer marker")
    return body[: at + len(marker)]


@torch.inference_mode()
def greedy_answer(
    model: FullContextTransformer,
    ex: Example,
    *,
    max_new_tokens: int = 20,
) -> str:
    ids = [BOS] + list(prompt_prefix(ex))
    out: List[int] = []
    for _ in range(max_new_tokens):
        tokens = torch.tensor([ids], dtype=torch.long)
        logits = model(tokens)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        if nxt == PAD or not (0 <= nxt < 256):
            break
        out.append(nxt)
        ids.append(nxt)
    return bytes(out).decode("utf-8", errors="replace")


@torch.inference_mode()
def evaluate_group(
    model: FullContextTransformer,
    examples: Sequence[Example],
    *,
    max_new_tokens: int = 20,
):
    correct = 0
    samples = []
    by_gold: Dict[str, Dict[str, int]] = {}
    for ex in examples:
        pred = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
        hit = pred == ex.answer
        correct += int(hit)
        row = by_gold.setdefault(ex.answer, {"correct": 0, "total": 0})
        row["correct"] += int(hit)
        row["total"] += 1
        if len(samples) < 20:
            samples.append({"task": ex.task, "gold": ex.answer, "pred": pred})
    return {
        "exact_accuracy": correct / max(1, len(examples)),
        "correct": correct,
        "n": len(examples),
        "by_gold": {
            k: {**v, "accuracy": v["correct"] / max(1, v["total"])}
            for k, v in by_gold.items()
        },
        "examples": samples,
    }


def task_groups(seed: int, n: int, hard: bool):
    out = {}
    for i, fn in enumerate(PROCESS_TASKS):
        rng = random.Random(seed + i * 100003)
        out[fn.__name__] = [fn(rng, hard).example for _ in range(n)]
    return out


def exact_balanced_graphs(seed: int, hard: bool, per_label: int):
    rng = random.Random(seed)
    buckets = {"yes": [], "no": []}
    attempts = 0
    while min(len(buckets["yes"]), len(buckets["no"])) < per_label:
        ex = graph_process(rng, hard).example
        if len(buckets[ex.answer]) < per_label:
            buckets[ex.answer].append(ex)
        attempts += 1
        if attempts > per_label * 200:
            raise RuntimeError("could not construct balanced graph set")
    rows = buckets["yes"] + buckets["no"]
    random.Random(seed + 991).shuffle(rows)
    return rows


def stress_groups(seed: int, n: int):
    specs: List[tuple[str, Callable[[random.Random], Example]]] = [
        ("state_12", lambda r: corrected_state_long(r, 12)),
        ("state_24", lambda r: corrected_state_long(r, 24)),
        ("state_48", lambda r: corrected_state_long(r, 48)),
        ("state_96", lambda r: corrected_state_long(r, 96)),
        ("assoc_12", lambda r: associative_long(r, 12)),
        ("assoc_24", lambda r: associative_long(r, 24)),
        ("assoc_48", lambda r: associative_long(r, 48)),
        ("assoc_96", lambda r: associative_long(r, 96)),
    ]
    out = {}
    for i, (name, fn) in enumerate(specs):
        rng = random.Random(seed + i * 100003)
        out[name] = [fn(rng) for _ in range(n)]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--mode", choices=["task", "balanced-graph", "stress"], required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261301)
    p.add_argument("--hard", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()

    model = load_model(args.checkpoint)
    if args.mode == "task":
        groups = task_groups(args.seed, args.n, args.hard)
        result = {
            name: evaluate_group(model, rows, max_new_tokens=args.max_new_tokens)
            for name, rows in groups.items()
        }
    elif args.mode == "balanced-graph":
        result = {}
        for hard in (False, True):
            rows = exact_balanced_graphs(
                args.seed + int(hard) * 99991, hard, args.n
            )
            result["hard" if hard else "easy"] = evaluate_group(
                model, rows, max_new_tokens=args.max_new_tokens
            )
    else:
        groups = stress_groups(args.seed, args.n)
        result = {
            name: evaluate_group(model, rows, max_new_tokens=args.max_new_tokens)
            for name, rows in groups.items()
        }

    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
