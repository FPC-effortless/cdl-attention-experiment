from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import EOS, PAD
from .full_context_baseline import FullContextTransformer, baseline_config
from .process_data import make_process_batch


def answer_supervision_mask(
    tokens: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise answer bytes plus the terminal EOS target for each row."""
    targets = tokens[:, 1:]
    return (answer_mask[:, 1:] | (targets == EOS)) & (targets != PAD)


def answer_only_loss(
    model: FullContextTransformer,
    tokens: torch.Tensor,
    answer_mask: torch.Tensor,
):
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = model(input_ids)
    mask = answer_supervision_mask(tokens, answer_mask)
    tok_nll = F.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=PAD, reduction="none"
    )
    denom = mask.sum().clamp_min(1)
    loss = (tok_nll * mask.to(tok_nll.dtype)).sum() / denom
    pred = logits.argmax(dim=-1)
    acc = (
        ((pred == targets) & mask).sum().to(torch.float32) / denom.to(torch.float32)
    )
    return loss, acc


@torch.inference_mode()
def evaluate(
    model: FullContextTransformer,
    *,
    seed: int,
    batches: int,
    batch_size: int,
    seq_len: int,
    hard: bool,
):
    model.eval()
    losses = []
    accs = []
    for i in range(batches):
        tokens, _, answer_mask, _ = make_process_batch(
            batch_size,
            seq_len,
            seed + i * 7919,
            hard=hard,
            reasoning_steps=3,
        )
        loss, acc = answer_only_loss(model, tokens, answer_mask)
        losses.append(float(loss))
        accs.append(float(acc))
    model.train()
    return {
        "answer_nll": sum(losses) / max(1, len(losses)),
        "answer_byte_accuracy": sum(accs) / max(1, len(accs)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="full-context-baseline-output")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=385)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261231)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=400)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    cfg = baseline_config()
    model = FullContextTransformer(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    history = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(
            1.0, step / max(1, args.hard_ramp_steps)
        )
        hard = random.Random(args.seed + step * 65537).random() < hard_prob
        tokens, _, answer_mask, _ = make_process_batch(
            args.batch_size,
            args.seq_len,
            args.seed + step * 104729,
            hard=hard,
            reasoning_steps=3,
        )
        loss, acc = answer_only_loss(model, tokens, answer_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "answer_nll": float(loss.detach()),
                "answer_byte_accuracy": float(acc.detach()),
                "hard_batch": bool(hard),
            }
            history.append(row)
            print(row, flush=True)

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
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(kind="full-context-answer-only"), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "model": "full-context-answer-only",
        "parameters": model.parameter_count(),
        "steps": args.steps,
        "seconds": elapsed,
        **{f"easy_{k}": v for k, v in easy.items()},
        **{f"hard_{k}": v for k, v in hard.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
