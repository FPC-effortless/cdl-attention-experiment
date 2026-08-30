from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F

from .answer_state import CASMAnswerState, answer_targets, decode_answer_logits
from .data import BOS, EOS, PAD, Example
from .model import CASMConfig
from .process_data import (
    PROCESS_TASKS,
    associative_long,
    corrected_state_long,
    graph_process,
)


def load_model(path: str, override_steps: int | None = None) -> CASMAnswerState:
    ckpt = torch.load(path, map_location="cpu")
    cfg = CASMConfig(**ckpt["config"])
    steps = int(override_steps if override_steps is not None else ckpt["reasoning_steps"])
    model = CASMAnswerState(
        cfg,
        reasoning_steps=steps,
        answer_slots=int(ckpt.get("answer_slots", 20)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def pack_examples(examples: Sequence[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[int]] = []
    anchors: List[int] = []
    marker = b"answer "
    max_len = 0
    for ex in examples:
        body = ex.text.encode("utf-8", errors="replace")
        at = body.rfind(marker)
        if at < 0:
            raise ValueError("example has no answer marker")
        anchor = 1 + at + len(marker) - 1
        row = [BOS] + list(body) + [EOS]
        max_len = max(max_len, len(row))
        rows.append(row)
        anchors.append(anchor)
    for row in rows:
        row.extend([PAD] * (max_len - len(row)))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(anchors, dtype=torch.long)


@torch.inference_mode()
def score_examples(model, examples: Sequence[Example], *, batch_size: int = 20):
    total = 0
    correct = 0
    nll_sum = 0.0
    step_sums = [0.0] * model.reasoning_steps
    sample_rows = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        toks, anchors = pack_examples(batch)
        targets = answer_targets(
            [x.answer for x in batch], model.answer_slots, device=toks.device
        )
        out = model(toks, anchors)
        preds = decode_answer_logits(out.logits_steps[-1])
        for ex, pred in zip(batch, preds):
            correct += int(pred == ex.answer)
            if len(sample_rows) < 16:
                sample_rows.append({"task": ex.task, "gold": ex.answer, "pred": pred})
        final_nll = F.cross_entropy(
            out.logits_steps[-1].transpose(1, 2), targets, ignore_index=PAD
        )
        nll_sum += float(final_nll) * len(batch)
        for i, logits in enumerate(out.logits_steps):
            step_sums[i] += float(
                F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD)
            ) * len(batch)
        total += len(batch)
    result = {
        "exact_accuracy": correct / max(1, total),
        "correct": correct,
        "n": total,
        "answer_nll": nll_sum / max(1, total),
        "examples": sample_rows,
    }
    for i, value in enumerate(step_sums, start=1):
        result[f"step{i}_nll"] = value / max(1, total)
    return result


def task_groups(seed: int, n: int, hard: bool):
    groups = {}
    for i, fn in enumerate(PROCESS_TASKS):
        rng = random.Random(seed + i * 100003)
        groups[fn.__name__] = [fn(rng, hard).example for _ in range(n)]
    return groups


def exact_balanced_graphs(seed: int, hard: bool, per_label: int) -> List[Example]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Example]] = {"yes": [], "no": []}
    attempts = 0
    while min(len(buckets["yes"]), len(buckets["no"])) < per_label:
        ex = graph_process(rng, hard).example
        if len(buckets[ex.answer]) < per_label:
            buckets[ex.answer].append(ex)
        attempts += 1
        if attempts > per_label * 100:
            raise RuntimeError("could not construct balanced graph set")
    rows = buckets["yes"] + buckets["no"]
    random.Random(seed + 999).shuffle(rows)
    return rows


def balanced_graph(model, seed: int, per_label: int):
    result = {}
    for hard in (False, True):
        rows = exact_balanced_graphs(seed + int(hard) * 99991, hard, per_label)
        scored = score_examples(model, rows)
        by_gold = {}
        for label in ("yes", "no"):
            by_gold[label] = score_examples(model, [x for x in rows if x.answer == label])
        scored["by_gold"] = by_gold
        result["hard" if hard else "easy"] = scored
    return result


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
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--mode", choices=["task", "balanced-graph", "stress", "depth"], required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261222)
    p.add_argument("--hard", action="store_true")
    p.add_argument("--depths", default="1,2,3,5")
    args = p.parse_args()

    result = {}
    for cp in args.checkpoints:
        base = Path(cp).stem
        if args.mode == "depth":
            depth_result = {}
            groups = task_groups(args.seed, args.n, True)
            flat = [x for rows in groups.values() for x in rows]
            for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
                model = load_model(cp, override_steps=depth)
                depth_result[str(depth)] = score_examples(model, flat)
            result[base] = depth_result
            continue

        model = load_model(cp)
        if args.mode == "task":
            groups = task_groups(args.seed, args.n, args.hard)
            result[base] = {name: score_examples(model, rows) for name, rows in groups.items()}
        elif args.mode == "balanced-graph":
            result[base] = balanced_graph(model, args.seed, args.n)
        else:
            groups = stress_groups(args.seed, args.n)
            result[base] = {name: score_examples(model, rows) for name, rows in groups.items()}
        print(base, args.mode, "complete", flush=True)

    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
