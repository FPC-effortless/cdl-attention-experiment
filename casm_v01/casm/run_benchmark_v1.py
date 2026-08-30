from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from .benchmark_v1 import VERSION, build_suite, contamination_report, score_predictions
from .data import Example
from .eval_recurrent import load_recurrent
from .eval_tasks import score_example
from .free_generate_eval import greedy_answer


def _as_example(case) -> Example:
    return Example(case.prompt + case.answer, case.task, case.answer)


def evaluate_checkpoint(checkpoint: str, suite_name: str, max_new_tokens: int, diagnostics: bool):
    model = load_recurrent(checkpoint)
    rows = build_suite(suite_name)
    preds: Dict[str, str] = {}
    diag = {}
    started = time.perf_counter()
    for i, case in enumerate(rows):
        ex = _as_example(case)
        preds[case.case_id] = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
        if diagnostics:
            diag[case.case_id] = score_example(model, ex)
        if (i + 1) % 100 == 0:
            print(Path(checkpoint).stem, suite_name, i + 1, "/", len(rows), flush=True)
    elapsed = time.perf_counter() - started
    result = score_predictions(rows, preds)
    result["model_diagnostics"] = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "reasoning_steps": int(getattr(model, "reasoning_steps", 1)),
        "evaluation_seconds": elapsed,
        "cases_per_second": len(rows) / elapsed if elapsed > 0 else None,
        "max_new_tokens": max_new_tokens,
    }
    if diagnostics:
        by_task = {}
        for task in sorted({x.task for x in rows}):
            vals = [diag[x.case_id] for x in rows if x.task == task]
            by_task[task] = {
                "answer_nll": sum(v["answer_nll"] for v in vals) / len(vals),
                "answer_byte_acc_tf": sum(v["answer_byte_acc"] for v in vals) / len(vals),
                "answer_exact_tf": sum(v["answer_exact_tf"] for v in vals) / len(vals),
            }
        result["diagnostic_teacher_forced"] = by_task
    return result, preds


def main() -> None:
    p = argparse.ArgumentParser(description=f"Canonical {VERSION} evaluator")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--suite", choices=["dev-core", "dev-ood", "holdout-core", "holdout-ood"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--allow-holdout", action="store_true", help="Required to access a HOLDOUT suite")
    p.add_argument("--train-hashes", help="Newline-delimited SHA256 prompt hashes emitted during training")
    p.add_argument("--out", default="benchmark-v1-result.json")
    args = p.parse_args()

    is_holdout = args.suite.startswith("holdout")
    if is_holdout and not args.allow_holdout:
        raise SystemExit("HOLDOUT is promotion-only. Re-run with --allow-holdout after freezing the candidate.")

    rows = build_suite(args.suite)
    contamination = None
    if args.train_hashes:
        contamination = contamination_report(rows, Path(args.train_hashes).read_text().splitlines())
    elif is_holdout:
        contamination = {"certified_clean": False, "reason": "no training prompt hash log supplied"}

    output = {"benchmark_version": VERSION, "suite": args.suite, "contamination": contamination, "models": {}}
    pred_dir = Path(args.out).with_suffix("").with_name(Path(args.out).stem + "-predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    for cp in args.checkpoints:
        name = Path(cp).stem
        result, preds = evaluate_checkpoint(cp, args.suite, args.max_new_tokens, args.diagnostics)
        output["models"][name] = result
        with (pred_dir / f"{name}.jsonl").open("w") as f:
            for case in rows:
                f.write(json.dumps({"case_id": case.case_id, "task": case.task, "prediction": preds[case.case_id], "gold": case.answer}, sort_keys=True) + "\n")
        print(name, json.dumps({
            "raw_solve_macro": result["raw_solve_macro"],
            "normalized_solve_macro": result["normalized_solve_macro"],
            "parameter_count": result["model_diagnostics"]["parameter_count"],
            "reasoning_steps": result["model_diagnostics"]["reasoning_steps"],
            "cases_per_second": result["model_diagnostics"]["cases_per_second"],
        }), flush=True)

    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True))
    print("RESULT_JSON")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
