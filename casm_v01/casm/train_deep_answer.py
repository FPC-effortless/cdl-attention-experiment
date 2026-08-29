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

from .data import PAD, VOCAB_SIZE
from .deep_answer import deep_answer_loss
from .model import CASMConfig
from .process_data import make_process_batch
from .recurrent_model import CASMRecurrent


def build_core(cfg: CASMConfig) -> CASMRecurrent:
    common = replace(
        cfg,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    return CASMRecurrent(common, reasoning_steps=3)


@torch.inference_mode()
def evaluate(
    model: CASMRecurrent,
    seed: int,
    batches: int,
    batch_size: int,
    seq_len: int,
    hard: bool,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    answer_nll: List[float] = []
    answer_acc: List[float] = []
    for i in range(batches):
        toks, _, answer_mask, _ = make_process_batch(
            batch_size,
            seq_len,
            seed + i * 7919,
            hard=hard,
            reasoning_steps=3,
        )
        out = model(toks, return_aux=False)
        losses.append(float(out["lm_loss"]))
        target = toks[:, 1:]
        mask = answer_mask[:, 1:] & (target != PAD)
        tok_nll = F.cross_entropy(
            out["logits"].transpose(1, 2), target, ignore_index=PAD, reduction="none"
        )
        pred = out["logits"].argmax(dim=-1)
        if mask.any():
            answer_nll.append(float(tok_nll[mask].mean()))
            answer_acc.append(float((pred[mask] == target[mask]).float().mean()))
    model.train()
    return {
        "lm_loss": sum(losses) / len(losses),
        "answer_nll": sum(answer_nll) / max(1, len(answer_nll)),
        "answer_byte_acc": sum(answer_acc) / max(1, len(answer_acc)),
    }


def train_one(
    kind: str,
    args,
    cfg: CASMConfig,
    init_core: Dict[str, torch.Tensor],
) -> Tuple[CASMRecurrent, List[Dict[str, float]], float]:
    model = build_core(cfg)
    model.load_state_dict(init_core)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(1.0, step / max(1, args.hard_ramp_steps))
        use_hard = random.Random(args.seed + step * 65537).random() < hard_prob
        toks, _, answer_mask, _ = make_process_batch(
            args.batch_size,
            args.seq_len,
            args.seed + step * 104729,
            hard=use_hard,
            reasoning_steps=3,
        )
        target_weights = 1.0 + (args.answer_weight - 1.0) * answer_mask[:, 1:].float()
        out = model(toks, target_weights=target_weights)

        deep_loss = out["loss"].new_zeros(())
        deep_metrics: Dict[str, torch.Tensor] = {}
        deep_weight = 0.0
        if kind == "deep-answer-3step" and step >= args.deep_start:
            deep_loss, deep_metrics = deep_answer_loss(model, toks, answer_mask)
            progress = (step - args.deep_start + 1) / max(1, args.deep_warmup)
            deep_weight = args.deep_weight * min(1.0, progress)

        total = out["loss"] + deep_weight * deep_loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "lm_loss": float(out["lm_loss"].detach()),
                "total_loss": float(total.detach()),
                "deep_answer_loss": float(deep_loss.detach()),
                "deep_weight": float(deep_weight),
                "deep_answer_acc": float(deep_metrics.get("deep_answer_acc", out["loss"].new_zeros(()))),
                "deep_answer_nll": float(deep_metrics.get("deep_answer_nll", out["loss"].new_zeros(()))),
                "router_entropy": float(out["router_entropy"].detach()),
                "memory_gate_mean": float(out["memory_gate_mean"].detach()),
            }
            for k in ("step1_answer_nll", "step2_answer_nll", "step1_answer_acc", "step2_answer_acc"):
                if k in deep_metrics:
                    row[k] = float(deep_metrics[k])
            history.append(row)
            print(kind, row, flush=True)

    return model, history, time.perf_counter() - started


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="casm-d-output")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=385)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261221)
    p.add_argument("--answer-weight", type=float, default=12.0)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=800)
    p.add_argument("--deep-start", type=int, default=200)
    p.add_argument("--deep-warmup", type=int, default=400)
    p.add_argument("--deep-weight", type=float, default=0.35)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--models", default="final-only-3step,deep-answer-3step")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    cfg = CASMConfig(vocab_size=VOCAB_SIZE)
    if args.tiny:
        cfg = replace(
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
        )

    probe = build_core(cfg)
    init_core = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for kind in [x.strip() for x in args.models.split(",") if x.strip()]:
        if kind not in {"final-only-3step", "deep-answer-3step"}:
            raise ValueError(kind)
        model, history, elapsed = train_one(kind, args, cfg, init_core)
        easy = evaluate(model, args.seed + 9_000_000, args.eval_batches, args.batch_size, args.seq_len, False)
        hard = evaluate(model, args.seed + 19_000_000, args.eval_batches, args.batch_size, args.seq_len, True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(model.cfg),
                "reasoning_steps": 3,
                "kind": kind,
            },
            out_dir / f"{kind}.pt",
        )
        (out_dir / f"{kind}-history.json").write_text(json.dumps(history, indent=2))
        summary.append(
            {
                "model": kind,
                "deployed_parameters": model.parameter_count(),
                "training_extra_parameters": 0,
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
