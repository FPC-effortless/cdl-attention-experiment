from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .data import EOS
from .eval_answer_state import load_model, pack_examples
from .process_data import exact_balanced_graphs if False else graph_process, state_process


STATE_CANDIDATES = ["bag", "box", "desk", "room", "shelf", "tray"]
GRAPH_CANDIDATES = ["yes", "no"]


def candidate_ids(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="replace")) + [EOS]


@torch.inference_mode()
def score_batch(model, examples, candidates: Sequence[str]):
    toks, anchors = pack_examples(examples)
    logits = model(toks, anchors).logits_steps[-1]
    logp = F.log_softmax(logits, dim=-1)
    predictions_sum: List[str] = []
    predictions_mean: List[str] = []
    margins_mean: List[float] = []

    for bi in range(len(examples)):
        sum_scores = []
        mean_scores = []
        for candidate in candidates:
            ids = candidate_ids(candidate)
            vals = torch.stack([logp[bi, pos, tok] for pos, tok in enumerate(ids)])
            sum_scores.append(float(vals.sum()))
            mean_scores.append(float(vals.mean()))
        order = sorted(range(len(candidates)), key=lambda j: mean_scores[j], reverse=True)
        predictions_sum.append(candidates[max(range(len(candidates)), key=lambda j: sum_scores[j])])
        predictions_mean.append(candidates[order[0]])
        margins_mean.append(mean_scores[order[0]] - mean_scores[order[1]] if len(order) > 1 else 0.0)

    return predictions_sum, predictions_mean, margins_mean


def evaluate(model, examples, candidates: Sequence[str], batch_size: int = 32) -> Dict[str, object]:
    correct_sum = 0
    correct_mean = 0
    total = 0
    margins = []
    by_gold: Dict[str, Dict[str, int]] = {}
    samples = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        pred_sum, pred_mean, batch_margins = score_batch(model, batch, candidates)
        for ex, ps, pm, margin in zip(batch, pred_sum, pred_mean, batch_margins):
            hit_sum = ps == ex.answer
            hit_mean = pm == ex.answer
            correct_sum += int(hit_sum)
            correct_mean += int(hit_mean)
            total += 1
            margins.append(margin)
            row = by_gold.setdefault(ex.answer, {"total": 0, "sum_correct": 0, "mean_correct": 0})
            row["total"] += 1
            row["sum_correct"] += int(hit_sum)
            row["mean_correct"] += int(hit_mean)
            if len(samples) < 20:
                samples.append({"gold": ex.answer, "sum_pred": ps, "mean_pred": pm, "mean_margin": margin})

    return {
        "n": total,
        "sum_logprob_accuracy": correct_sum / max(1, total),
        "mean_logprob_accuracy": correct_mean / max(1, total),
        "sum_logprob_correct": correct_sum,
        "mean_logprob_correct": correct_mean,
        "mean_top2_margin": sum(margins) / max(1, len(margins)),
        "by_gold": {
            gold: {
                **vals,
                "sum_accuracy": vals["sum_correct"] / vals["total"],
                "mean_accuracy": vals["mean_correct"] / vals["total"],
            }
            for gold, vals in by_gold.items()
        },
        "examples": samples,
    }


def balanced_graph_examples(seed: int, hard: bool, per_label: int):
    rng = random.Random(seed)
    buckets = {"yes": [], "no": []}
    attempts = 0
    while min(len(buckets["yes"]), len(buckets["no"])) < per_label:
        ex = graph_process(rng, hard).example
        if len(buckets[ex.answer]) < per_label:
            buckets[ex.answer].append(ex)
        attempts += 1
        if attempts > per_label * 200:
            raise RuntimeError("could not build balanced graph set")
    rows = buckets["yes"] + buckets["no"]
    random.Random(seed + 991).shuffle(rows)
    return rows


def state_examples(seed: int, hard: bool, n: int):
    rng = random.Random(seed)
    return [state_process(rng, hard).example for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261226)
    args = p.parse_args()

    graph_easy = balanced_graph_examples(args.seed, False, args.n // 2)
    graph_hard = balanced_graph_examples(args.seed + 1, True, args.n // 2)
    state_easy = state_examples(args.seed + 2, False, args.n)
    state_hard = state_examples(args.seed + 3, True, args.n)

    result = {}
    for cp in args.checkpoints:
        model = load_model(cp)
        result[cp] = {
            "graph_easy": evaluate(model, graph_easy, GRAPH_CANDIDATES),
            "graph_hard": evaluate(model, graph_hard, GRAPH_CANDIDATES),
            "state_easy": evaluate(model, state_easy, STATE_CANDIDATES),
            "state_hard": evaluate(model, state_hard, STATE_CANDIDATES),
        }
        print(cp, "complete", flush=True)

    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
