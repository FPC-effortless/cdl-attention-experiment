from __future__ import annotations

import argparse
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import torch

from .data import BOS, EOS, Example
from .eval_answer_state import (
    load_model as load_answer_state,
    pack_examples as pack_answer_state_examples,
    score_examples as score_answer_state_examples,
)
from .eval_process import exact_balanced_graphs
from .eval_recurrent import load_recurrent
from .eval_tasks import load_model as load_base, prefix_answer, score_example
from .free_generate_eval import greedy_answer
from .process_data import PROCESS_TASKS, associative_long, corrected_state_long

HARNESS_VERSION = "casm-unified-v2-2026-08-31"
TASK_SEED = 2026083101
BALANCED_GRAPH_SEED = 2026083102
STRESS_SEED = 2026083103
PROFILE_SEED = 2026083104


def mean_dict(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in rows[0]}


def task_groups(seed: int, n: int, hard: bool) -> Dict[str, List[Example]]:
    groups: Dict[str, List[Example]] = {}
    for index, fn in enumerate(PROCESS_TASKS):
        rng = random.Random(seed + index * 100003 + int(hard) * 10000019)
        groups[fn.__name__] = [fn(rng, hard).example for _ in range(n)]
    return groups


def stress_groups(seed: int, n: int) -> Dict[str, List[Example]]:
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
    out: Dict[str, List[Example]] = {}
    for index, (name, fn) in enumerate(specs):
        rng = random.Random(seed + index * 100003)
        out[name] = [fn(rng) for _ in range(n)]
    return out


def score_generative_model(model, examples: Sequence[Example], max_new_tokens: int) -> Dict[str, object]:
    likelihood = [score_example(model, ex) for ex in examples]
    exact = 0
    samples: List[Dict[str, str]] = []
    by_gold: Dict[str, Dict[str, int]] = {}
    for index, ex in enumerate(examples):
        pred = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
        hit = pred == ex.answer.strip()
        exact += int(hit)
        bucket = by_gold.setdefault(ex.answer.strip(), {"correct": 0, "total": 0})
        bucket["correct"] += int(hit)
        bucket["total"] += 1
        if index < 12:
            samples.append({"task": ex.task, "gold": ex.answer.strip(), "pred": pred})
    return {
        **mean_dict(likelihood),
        "exact_accuracy": exact / max(1, len(examples)),
        "exact_correct": exact,
        "n": len(examples),
        "by_gold": {
            label: {**vals, "accuracy": vals["correct"] / max(1, vals["total"])}
            for label, vals in sorted(by_gold.items())
        },
        "examples": samples,
    }


def score_answer_state_model(model, examples: Sequence[Example]) -> Dict[str, object]:
    return score_answer_state_examples(model, examples)


def evaluate_capability(family: str, model, n_per_task: int, stress_n: int, graph_per_label: int, max_new_tokens: int) -> Dict[str, object]:
    result: Dict[str, object] = {"easy": {}, "hard": {}, "stress": {}, "balanced_graph": {}}
    scorer = (
        (lambda rows: score_answer_state_model(model, rows))
        if family == "answer-state"
        else (lambda rows: score_generative_model(model, rows, max_new_tokens))
    )

    for hard in (False, True):
        groups = task_groups(TASK_SEED, n_per_task, hard)
        bucket: Dict[str, object] = {}
        flat: List[Example] = []
        for name, rows in groups.items():
            bucket[name] = scorer(rows)
            flat.extend(rows)
        bucket["overall"] = scorer(flat)
        result["hard" if hard else "easy"] = bucket

    for name, rows in stress_groups(STRESS_SEED, stress_n).items():
        result["stress"][name] = scorer(rows)

    for hard in (False, True):
        rows = exact_balanced_graphs(
            BALANCED_GRAPH_SEED + int(hard) * 99991,
            hard,
            graph_per_label,
        )
        result["balanced_graph"]["hard" if hard else "easy"] = scorer(rows)

    return result


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / max(1, iters)


def peak_bytes(fn) -> int:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        fn()
        sync()
        return int(torch.cuda.max_memory_allocated())
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def random_tokens(batch: int, seq_len: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, seq_len), generator=g, dtype=torch.long)


def run_forward(family: str, model, tokens: torch.Tensor):
    if family == "answer-state":
        anchors = torch.full((tokens.shape[0],), tokens.shape[1] - 2, dtype=torch.long)
        return model(tokens, anchors)
    return model(tokens, return_aux=False)


def run_decode_recompute(model, seq_len: int, batch: int, steps: int, seed: int) -> None:
    prompt = random_tokens(batch, seq_len, seed)
    generated: List[torch.Tensor] = []
    for _ in range(steps):
        if generated:
            body = torch.cat([prompt] + generated, dim=1)
        else:
            body = prompt
        dummy = torch.full((batch, 1), EOS, dtype=torch.long)
        toks = torch.cat([body, dummy], dim=1)
        out = model(toks, return_aux=False)
        nxt = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
        nxt = torch.remainder(nxt, 256)
        generated.append(nxt)


def profile_efficiency(
    family: str,
    model,
    seq_lens: Iterable[int],
    batch: int,
    decode_steps: int,
    warmup: int,
    iters: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    param_count = sum(p.numel() for p in model.parameters())
    for seq_len in seq_lens:
        tokens = random_tokens(batch, seq_len, PROFILE_SEED + seq_len)
        forward_fn = lambda: run_forward(family, model, tokens)
        prefill_s = timed(forward_fn, warmup, iters)
        prefill_peak = peak_bytes(forward_fn)
        row: Dict[str, object] = {
            "seq_len": seq_len,
            "batch_size": batch,
            "parameters": param_count,
            "prefill_seconds": prefill_s,
            "prefill_tokens_per_second": batch * seq_len / max(prefill_s, 1e-12),
            "prefill_peak_memory_bytes": prefill_peak,
        }
        if family == "answer-state":
            answer_slots = int(getattr(model, "answer_slots", 20))
            row.update(
                {
                    "decode_mode": "parallel_answer_state",
                    "decode_steps": answer_slots,
                    "decode_seconds": prefill_s,
                    "decode_tokens_per_second": batch * answer_slots / max(prefill_s, 1e-12),
                    "decode_peak_memory_bytes": prefill_peak,
                    "carried_query_supported": False,
                }
            )
        else:
            decode_fn = lambda: run_decode_recompute(
                model,
                seq_len=seq_len,
                batch=batch,
                steps=decode_steps,
                seed=PROFILE_SEED + seq_len + 17,
            )
            decode_s = timed(decode_fn, max(1, warmup // 2), iters)
            decode_peak = peak_bytes(decode_fn)
            row.update(
                {
                    "decode_mode": "full_context_recompute",
                    "decode_steps": decode_steps,
                    "decode_seconds": decode_s,
                    "decode_tokens_per_second": batch * decode_steps / max(decode_s, 1e-12),
                    "decode_peak_memory_bytes": decode_peak,
                    "carried_query_supported": False,
                }
            )
        rows.append(row)
    return rows


def load_checkpoint(family: str, path: str, override_steps: int | None = None):
    if family == "base":
        return load_base(path)
    if family == "recurrent":
        return load_recurrent(path, override_steps=override_steps)
    if family == "answer-state":
        return load_answer_state(path, override_steps=override_steps)
    raise ValueError(family)


def apply_control_ratios(models: Dict[str, Dict[str, object]], control: str | None) -> None:
    if not control or control not in models:
        return
    control_rows = {int(row["seq_len"]): row for row in models[control]["efficiency"]}
    for name, payload in models.items():
        for row in payload["efficiency"]:
            base = control_rows[int(row["seq_len"])]
            row["prefill_vs_control"] = float(row["prefill_tokens_per_second"]) / max(float(base["prefill_tokens_per_second"]), 1e-12)
            row["decode_vs_control"] = float(row["decode_tokens_per_second"]) / max(float(base["decode_tokens_per_second"]), 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["base", "recurrent", "answer-state"], required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--control", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-per-task", type=int, default=60)
    p.add_argument("--stress-n", type=int, default=30)
    p.add_argument("--graph-per-label", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--profile-seq-lens", default="96,192,384")
    p.add_argument("--profile-batch", type=int, default=4)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    seq_lens = [int(x) for x in args.profile_seq_lens.split(",") if x.strip()]
    models: Dict[str, Dict[str, object]] = {}
    for cp in args.checkpoints:
        name = Path(cp).stem
        model = load_checkpoint(args.family, cp)
        models[name] = {
            "checkpoint": str(cp),
            "family": args.family,
            "capability": evaluate_capability(
                args.family,
                model,
                n_per_task=args.n_per_task,
                stress_n=args.stress_n,
                graph_per_label=args.graph_per_label,
                max_new_tokens=args.max_new_tokens,
            ),
            "efficiency": profile_efficiency(
                args.family,
                model,
                seq_lens=seq_lens,
                batch=args.profile_batch,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                iters=args.iters,
            ),
        }
        print(name, "harness complete", flush=True)

    apply_control_ratios(models, args.control or None)
    result = {
        "harness_version": HARNESS_VERSION,
        "contract": {
            "task_seed": TASK_SEED,
            "balanced_graph_seed": BALANCED_GRAPH_SEED,
            "stress_seed": STRESS_SEED,
            "profile_seed": PROFILE_SEED,
            "true_generation": args.family != "answer-state",
            "answer_state_no_gold_leakage": args.family == "answer-state",
            "balanced_graph_is_exact_50_50": True,
            "state_tracking_initial_state_is_explicit": True,
            "one_example_one_memory_lifetime_at_evaluation": True,
            "cpu_peak_memory": "tracemalloc proxy; CUDA uses max_memory_allocated",
            "decode_semantics": "full-context recompute unless model uses parallel answer-state slots",
        },
        "control": args.control or None,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
