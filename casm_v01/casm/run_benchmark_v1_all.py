from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .answer_state import answer_targets, decode_answer_logits
from .benchmark_v1 import VERSION, BenchCase, build_suite, score_predictions
from .data import Example, PAD
from .eval_answer_state import load_model as load_answer_state, pack_examples
from .eval_recurrent import load_recurrent
from .eval_tasks import load_model as load_base, score_example
from .free_generate_eval import greedy_answer
from .model import CASMConfig
from .selective import SelectiveCASM


def as_gold_example(case: BenchCase) -> Example:
    # greedy_answer discards everything after the literal marker before model
    # execution, so including gold here cannot leak it into generation.
    return Example(case.prompt + case.answer, case.task, case.answer)


def load_selective(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = CASMConfig(**ckpt["config"])
    model = SelectiveCASM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_family(family: str, checkpoint: str):
    if family == "base":
        return load_base(checkpoint)
    if family == "selective":
        return load_selective(checkpoint)
    if family == "recurrent":
        return load_recurrent(checkpoint)
    if family == "answer-state":
        return load_answer_state(checkpoint)
    raise ValueError(family)


@torch.inference_mode()
def answer_state_predictions(model, rows: Sequence[BenchCase], batch_size: int = 24) -> tuple[Dict[str, str], dict]:
    preds: Dict[str, str] = {}
    by_task_nll: Dict[str, list[float]] = {}
    by_task_exact: Dict[str, list[float]] = {}
    for start in range(0, len(rows), batch_size):
        batch = list(rows[start:start + batch_size])
        examples = [as_gold_example(case) for case in batch]
        tokens, anchors = pack_examples(examples)
        targets = answer_targets([case.answer for case in batch], model.answer_slots, device=tokens.device)
        out = model(tokens, anchors)
        logits = out.logits_steps[-1]
        decoded = decode_answer_logits(logits)
        per_token = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD, reduction="none")
        for i, (case, pred) in enumerate(zip(batch, decoded)):
            preds[case.case_id] = pred
            mask = targets[i] != PAD
            nll = float(per_token[i][mask].mean()) if mask.any() else 0.0
            by_task_nll.setdefault(case.task, []).append(nll)
            by_task_exact.setdefault(case.task, []).append(float(pred == case.answer))
    diag = {
        task: {
            "answer_nll": sum(vals) / len(vals),
            "native_exact": sum(by_task_exact[task]) / len(by_task_exact[task]),
        }
        for task, vals in by_task_nll.items()
    }
    return preds, diag


def evaluate_checkpoint(family: str, checkpoint: str, suite_name: str, max_new_tokens: int, diagnostics: bool):
    model = load_family(family, checkpoint)
    rows = build_suite(suite_name)
    started = time.perf_counter()
    diag = None

    if family == "answer-state":
        preds, diag = answer_state_predictions(model, rows)
    else:
        preds = {}
        teacher_diag: Dict[str, list[dict]] = {}
        for i, case in enumerate(rows):
            ex = as_gold_example(case)
            preds[case.case_id] = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
            if diagnostics:
                teacher_diag.setdefault(case.task, []).append(score_example(model, ex))
            if (i + 1) % 100 == 0:
                print(Path(checkpoint).stem, suite_name, i + 1, "/", len(rows), flush=True)
        if diagnostics:
            diag = {
                task: {
                    "answer_nll": sum(v["answer_nll"] for v in vals) / len(vals),
                    "answer_byte_acc_tf": sum(v["answer_byte_acc"] for v in vals) / len(vals),
                    "answer_exact_tf": sum(v["answer_exact_tf"] for v in vals) / len(vals),
                }
                for task, vals in teacher_diag.items()
            }

    elapsed = time.perf_counter() - started
    result = score_predictions(rows, preds)
    result["model_diagnostics"] = {
        "family": family,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "reasoning_steps": int(getattr(model, "reasoning_steps", 1)),
        "evaluation_seconds": elapsed,
        "cases_per_second": len(rows) / elapsed if elapsed > 0 else None,
        "max_new_tokens": max_new_tokens,
        "decode_protocol": (
            "parallel_answer_state_no_gold_length" if family == "answer-state"
            else "canonical_greedy_autoregressive"
        ),
        "canonical_decode_protocol": family != "answer-state",
    }
    if diag is not None:
        result["diagnostic_teacher_forced_or_native"] = diag
    return result, preds


def main() -> None:
    p = argparse.ArgumentParser(description=f"{VERSION} evaluator with checkpoint-family adapters")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--family", choices=["base", "selective", "recurrent", "answer-state"], required=True)
    p.add_argument("--suite", choices=["dev-core", "dev-ood", "holdout-core", "holdout-ood"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--allow-holdout", action="store_true")
    p.add_argument("--out", default="benchmark-v1-result.json")
    args = p.parse_args()

    if args.suite.startswith("holdout") and not args.allow_holdout:
        raise SystemExit("HOLDOUT is promotion-only; DEV must establish a promotion candidate first.")

    rows = build_suite(args.suite)
    output = {
        "benchmark_version": VERSION,
        "suite": args.suite,
        "family": args.family,
        "models": {},
    }
    pred_dir = Path(args.out).with_suffix("").with_name(Path(args.out).stem + "-predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for cp in args.checkpoints:
        name = Path(cp).stem
        result, preds = evaluate_checkpoint(args.family, cp, args.suite, args.max_new_tokens, args.diagnostics)
        output["models"][name] = result
        with (pred_dir / f"{name}.jsonl").open("w") as f:
            for case in rows:
                f.write(json.dumps({
                    "case_id": case.case_id,
                    "task": case.task,
                    "prediction": preds[case.case_id],
                    "gold": case.answer,
                }, sort_keys=True) + "\n")
        print(name, json.dumps({
            "raw_solve_macro": result["raw_solve_macro"],
            "normalized_solve_macro": result["normalized_solve_macro"],
            "parameter_count": result["model_diagnostics"]["parameter_count"],
            "reasoning_steps": result["model_diagnostics"]["reasoning_steps"],
            "canonical_decode_protocol": result["model_diagnostics"]["canonical_decode_protocol"],
        }), flush=True)

    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True))
    print("RESULT_JSON")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
