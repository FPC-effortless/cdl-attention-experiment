from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from .benchmark_v1 import VERSION, build_suite, score_predictions
from .data import BOS, EOS, Example
from .eval_tasks import load_model, score_example


def _example(case):
    return Example(case.prompt + case.answer, case.task, case.answer)


@torch.inference_mode()
def greedy_answer(model, case, max_new_tokens=64):
    prefix = case.prompt
    prefix_ids = list(prefix.encode("utf-8", errors="replace"))
    generated = []
    for _ in range(max_new_tokens):
        toks = torch.tensor([[BOS] + prefix_ids + generated + [EOS]], dtype=torch.long)
        out = model(toks, return_aux=False)
        nxt = int(out["logits"][0, -1].argmax())
        if nxt in (EOS, 10, 13) or not 0 <= nxt < 256:
            break
        generated.append(nxt)
    return bytes(generated).decode("utf-8", errors="replace").strip()


def evaluate(checkpoint, suite, max_new_tokens, diagnostics):
    model = load_model(checkpoint)
    rows = build_suite(suite)
    preds: Dict[str, str] = {}
    diag = {}
    started = time.perf_counter()
    for i, case in enumerate(rows):
        preds[case.case_id] = greedy_answer(model, case, max_new_tokens)
        if diagnostics:
            diag[case.case_id] = score_example(model, _example(case))
        if (i + 1) % 100 == 0:
            print(Path(checkpoint).stem, suite, i + 1, "/", len(rows), flush=True)
    elapsed = time.perf_counter() - started
    result = score_predictions(rows, preds)
    result["model_diagnostics"] = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "reasoning_steps": 1,
        "evaluation_seconds": elapsed,
        "cases_per_second": len(rows) / elapsed if elapsed else None,
        "max_new_tokens": max_new_tokens,
        "legacy_exact_source": True,
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
    return rows, result, preds


def main():
    p = argparse.ArgumentParser(description=f"Exact-source legacy adapter for {VERSION}")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--suite", choices=["dev-core", "dev-ood"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    output = {"benchmark_version": VERSION, "suite": args.suite, "models": {}}
    pred_dir = Path(args.out).with_suffix("").with_name(Path(args.out).stem + "-predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    for cp in args.checkpoints:
        rows, result, preds = evaluate(cp, args.suite, args.max_new_tokens, args.diagnostics)
        name = Path(cp).stem
        output["models"][name] = result
        with (pred_dir / f"{name}.jsonl").open("w") as f:
            for case in rows:
                f.write(json.dumps({"case_id":case.case_id,"task":case.task,"prediction":preds[case.case_id],"gold":case.answer}, sort_keys=True)+"\n")
        print(name, result["normalized_solve_macro"], result["raw_solve_macro"], flush=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
