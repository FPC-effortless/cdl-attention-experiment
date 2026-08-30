from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .answer_state import CASMAnswerState, answer_state_loss, answer_targets, decode_answer_logits
from .data import PAD, VOCAB_SIZE
from .model import CASMConfig
from .process_data import make_process_batch


def tiny_cfg() -> CASMConfig:
    cfg = CASMConfig(vocab_size=VOCAB_SIZE)
    return replace(
        cfg,
        d_model=96,
        n_layers=3,
        n_heads=4,
        n_kv_heads=1,
        d_ff=256,
        memory_dim=48,
        memory_slots=6,
        state_slots=2,
        chunk_size=24,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
        mtp_loss_weight=0.0,
        verifier_loss_weight=0.0,
    )


@torch.inference_mode()
def evaluate(model, *, seed, batches, batch_size, seq_len, hard):
    model.eval()
    nlls = []
    exact = 0
    total = 0
    step_nll = [[] for _ in range(model.reasoning_steps)]
    for i in range(batches):
        toks, pex, _, anchors = make_process_batch(
            batch_size, seq_len, seed + i * 7919, hard=hard, reasoning_steps=3
        )
        targets = answer_targets(
            [x.example.answer for x in pex], model.answer_slots, device=toks.device
        )
        out = model(toks, anchors)
        for si, logits in enumerate(out.logits_steps):
            step_nll[si].append(
                float(F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD))
            )
        final = out.logits_steps[-1]
        nlls.append(float(F.cross_entropy(final.transpose(1, 2), targets, ignore_index=PAD)))
        preds = decode_answer_logits(final)
        gold = [x.example.answer for x in pex]
        exact += sum(int(a == b) for a, b in zip(preds, gold))
        total += len(gold)
    model.train()
    result = {
        "answer_nll": sum(nlls) / len(nlls),
        "exact_accuracy": exact / max(1, total),
        "exact_correct": exact,
        "n": total,
    }
    for si, vals in enumerate(step_nll, start=1):
        result[f"step{si}_nll"] = sum(vals) / len(vals)
    return result


def train_one(kind, args, cfg, init_state):
    reasoning_steps = 1 if kind == "answer-state-1step" else 3
    model = CASMAnswerState(cfg, reasoning_steps=reasoning_steps, answer_slots=args.answer_slots)
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    history = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(1.0, step / max(1, args.hard_ramp_steps))
        hard = random.Random(args.seed + step * 65537).random() < hard_prob
        toks, pex, _, anchors = make_process_batch(
            args.batch_size,
            args.seq_len,
            args.seed + step * 104729,
            hard=hard,
            reasoning_steps=3,
        )
        targets = answer_targets(
            [x.example.answer for x in pex], args.answer_slots, device=toks.device
        )
        out = model(toks, anchors)
        loss, losses = answer_state_loss(out, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            preds = decode_answer_logits(out.logits_steps[-1])
            gold = [x.example.answer for x in pex]
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "batch_exact": sum(int(a == b) for a, b in zip(preds, gold)) / len(gold),
            }
            for i, v in enumerate(losses, start=1):
                row[f"step{i}_nll"] = float(v.detach())
            history.append(row)
            print(kind, row, flush=True)

    return model, history, time.perf_counter() - started


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="casm-y-output")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=385)
    p.add_argument("--answer-slots", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261221)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=800)
    p.add_argument("--models", default="answer-state-1step,answer-state-3step")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    cfg = tiny_cfg()

    probe = CASMAnswerState(cfg, reasoning_steps=3, answer_slots=args.answer_slots)
    init_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for kind in [x.strip() for x in args.models.split(",") if x.strip()]:
        if kind not in {"answer-state-1step", "answer-state-3step"}:
            raise ValueError(kind)
        model, history, elapsed = train_one(kind, args, cfg, init_state)
        easy = evaluate(
            model,
            seed=args.seed + 9_000_000,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hard=False,
        )
        hard = evaluate(
            model,
            seed=args.seed + 19_000_000,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hard=True,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "reasoning_steps": model.reasoning_steps,
                "answer_slots": model.answer_slots,
                "kind": kind,
            },
            out_dir / f"{kind}.pt",
        )
        (out_dir / f"{kind}-history.json").write_text(json.dumps(history, indent=2))
        summary.append(
            {
                "model": kind,
                "parameters": model.parameter_count(),
                "seconds": elapsed,
                **{f"easy_{k}": v for k, v in easy.items()},
                **{f"hard_{k}": v for k, v in hard.items()},
            }
        )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
