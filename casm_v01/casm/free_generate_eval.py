from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import torch

from .data import BOS, EOS, SEP, Example, graph_reachability
from .eval_tasks import load_model, prefix_answer
from .stress_eval import associative_long, state_long


def _decode_generated(ids: List[int]) -> str:
    raw = bytes([x for x in ids if 0 <= x < 256])
    return raw.decode("utf-8", errors="replace").strip()


@torch.inference_mode()
def greedy_answer(model, ex: Example, max_new_tokens: int = 32) -> str:
    """Generate an answer without teacher forcing or oracle answer length.

    Generation begins after the literal ``answer `` marker and terminates on
    EOS/SEP/newline or the fixed safety cap. The gold answer is never appended
    to the model input.
    """
    prefix, _ = prefix_answer(ex)
    prefix_ids = list(prefix.encode("utf-8", errors="replace"))
    generated: List[int] = []

    for _ in range(max_new_tokens):
        # The final EOS is a dummy target position. CASM consumes everything
        # except that final token and its last logit predicts the next token.
        tokens = torch.tensor([[BOS] + prefix_ids + generated + [EOS]], dtype=torch.long)
        out = model(tokens, return_aux=False)
        nxt = int(out["logits"][0, -1].argmax())
        if nxt in (EOS, SEP) or nxt in (10, 13):
            break
        if not 0 <= nxt < 256:
            break
        generated.append(nxt)

    return _decode_generated(generated)


def evaluate_group(
    model,
    fn: Callable[[random.Random], Example],
    n: int,
    seed: int,
    max_new_tokens: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    correct = 0
    examples: List[Dict[str, str]] = []
    by_label: Dict[str, Dict[str, int]] = {}

    for i in range(n):
        ex = fn(rng)
        pred = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
        gold = ex.answer.strip()
        hit = pred == gold
        correct += int(hit)
        bucket = by_label.setdefault(gold, {"correct": 0, "total": 0})
        bucket["correct"] += int(hit)
        bucket["total"] += 1
        if i < 12:
            examples.append({"gold": gold, "pred": pred})

    return {
        "exact_accuracy": correct / n,
        "correct": correct,
        "n": n,
        "by_gold": {
            label: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
            }
            for label, counts in sorted(by_label.items())
        },
        "examples": examples,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()

    groups: Dict[str, Callable[[random.Random], Example]] = {
        "graph_easy_balanced": lambda r: graph_reachability(r, hard=False),
        "graph_hard_balanced": lambda r: graph_reachability(r, hard=True),
        "state_12": lambda r: state_long(r, 12),
        "state_24": lambda r: state_long(r, 24),
        "state_48": lambda r: state_long(r, 48),
        "state_96": lambda r: state_long(r, 96),
        "assoc_12": lambda r: associative_long(r, 12),
        "assoc_24": lambda r: associative_long(r, 24),
        "assoc_48": lambda r: associative_long(r, 48),
        "assoc_96": lambda r: associative_long(r, 96),
    }

    result: Dict[str, object] = {}
    for cp in args.checkpoints:
        model = load_model(cp)
        name = Path(cp).stem
        result[name] = {}
        for i, (group, fn) in enumerate(groups.items()):
            metrics = evaluate_group(
                model,
                fn,
                n=args.n,
                seed=args.seed + i * 100003,
                max_new_tokens=args.max_new_tokens,
            )
            result[name][group] = metrics
            print(name, group, json.dumps({k: v for k, v in metrics.items() if k != "examples"}), flush=True)

    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
