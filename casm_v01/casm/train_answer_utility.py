from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import torch

from .answer_utility import answer_utility_qk_loss
from .data import VOCAB_SIZE, gzip_teacher_distributions, make_batch
from .model import CASM, CASMConfig
from .train import evaluate


def build_model(kind: str, cfg: CASMConfig) -> CASM:
    if kind == "qk-memory":
        return CASM(replace(
            cfg,
            use_compression_score=False,
            compression_loss_weight=0.0,
            compression_predictor_loss_weight=0.0,
        ))
    if kind == "compression-qk":
        return CASM(replace(cfg, use_compression_score=False))
    if kind == "answer-utility-qk":
        return CASM(replace(
            cfg,
            use_compression_score=False,
            compression_loss_weight=0.0,
            compression_predictor_loss_weight=0.0,
        ))
    raise ValueError(kind)


def train_one(kind: str, args, cfg: CASMConfig, init_state: Dict[str, torch.Tensor]):
    model = build_model(kind, cfg)
    own = model.state_dict()
    compatible = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
    own.update(compatible)
    model.load_state_dict(own)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    final_compression_weight = model.cfg.compression_loss_weight
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(
            1.0, step / max(1, args.hard_ramp_steps)
        )
        use_hard = random.Random(args.seed + step * 65537).random() < hard_prob
        toks, _, answer_mask = make_batch(
            args.batch_size,
            args.seq_len,
            args.seed + step * 104729,
            hard=use_hard,
            return_answer_mask=True,
        )
        target_weights = 1.0 + (
            args.answer_weight - 1.0
        ) * answer_mask[:, 1:].float()

        if kind == "compression-qk":
            model.cfg.compression_loss_weight = final_compression_weight * min(
                1.0, step / max(1, args.compression_warmup)
            )
            external = gzip_teacher_distributions(
                toks,
                model.cfg.chunk_size,
                model.cfg.memory_slots,
                model.cfg.state_slots,
                args.gzip_temperature,
            )
            decay = max(0.0, 1.0 - step / max(1, args.gzip_teacher_decay))
            alpha = args.gzip_teacher_floor + (
                1.0 - args.gzip_teacher_floor
            ) * decay
            out = model(
                toks,
                external_teacher=external,
                teacher_alpha=alpha,
                target_weights=target_weights,
            )
            utility = None
            utility_weight = 0.0
            total_loss = out["loss"]
        elif kind == "answer-utility-qk":
            out = model(toks, target_weights=target_weights)
            utility = answer_utility_qk_loss(
                model,
                toks,
                answer_mask[:, 1:],
                temperature=args.utility_temperature,
            )
            progress = max(0.0, step - args.utility_start) / max(
                1, args.utility_warmup
            )
            utility_weight = args.utility_weight * min(1.0, progress)
            total_loss = out["loss"] + utility_weight * utility["loss"]
        else:
            out = model(toks, target_weights=target_weights)
            utility = None
            utility_weight = 0.0
            total_loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                k: float(v.detach())
                for k, v in out.items()
                if k != "logits"
            }
            row["total_train_loss"] = float(total_loss.detach())
            row["answer_utility_weight"] = float(utility_weight)
            if utility is not None:
                row.update(
                    {
                        f"answer_utility_{k}": float(v.detach())
                        for k, v in utility.items()
                    }
                )
            row["step"] = step
            history.append(row)
            print(kind, row, flush=True)

    elapsed = time.perf_counter() - started
    easy = evaluate(
        model,
        cfg,
        args.seed + 9_000_000,
        args.eval_batches,
        args.batch_size,
        False,
    )
    hard = evaluate(
        model,
        cfg,
        args.seed + 19_000_000,
        args.eval_batches,
        args.batch_size,
        True,
    )
    return model, history, easy, hard, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="casm-u-output")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=289)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--answer-weight", type=float, default=8.0)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=600)
    p.add_argument("--compression-warmup", type=int, default=125)
    p.add_argument("--gzip-teacher-decay", type=int, default=800)
    p.add_argument("--gzip-temperature", type=float, default=1.0)
    p.add_argument("--gzip-teacher-floor", type=float, default=0.35)
    p.add_argument("--utility-start", type=int, default=200)
    p.add_argument("--utility-warmup", type=int, default=250)
    p.add_argument("--utility-weight", type=float, default=0.15)
    p.add_argument("--utility-temperature", type=float, default=0.25)
    p.add_argument("--tiny", action="store_true")
    p.add_argument(
        "--models",
        default="qk-memory,compression-qk,answer-utility-qk",
    )
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
        )

    probe = CASM(cfg)
    init = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    print("config", json.dumps(asdict(cfg), indent=2), flush=True)
    print("parameters", probe.parameter_count(), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for kind in [x.strip() for x in args.models.split(",") if x.strip()]:
        model, history, easy, hard, elapsed = train_one(
            kind, args, cfg, init
        )
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(model.cfg)},
            out_dir / f"{kind}.pt",
        )
        (out_dir / f"{kind}-history.json").write_text(
            json.dumps(history, indent=2)
        )
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
