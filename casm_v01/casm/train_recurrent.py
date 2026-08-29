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

from .data import VOCAB_SIZE, make_batch
from .model import CASMConfig
from .recurrent_model import CASMRecurrent
from .set_utility import set_conditioned_utility_loss
from .train import evaluate


def build_model(kind: str, cfg: CASMConfig) -> CASMRecurrent:
    common = replace(
        cfg,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    if kind == "qk-1step":
        return CASMRecurrent(common, reasoning_steps=1)
    if kind in ("qk-3step", "set-utility-3step"):
        return CASMRecurrent(common, reasoning_steps=3)
    raise ValueError(kind)


def train_one(
    kind: str,
    args,
    cfg: CASMConfig,
    init_state: Dict[str, torch.Tensor],
) -> Tuple[CASMRecurrent, List[Dict[str, float]], Dict[str, float], Dict[str, float], float]:
    model = build_model(kind, cfg)
    own = model.state_dict()
    compatible = {
        k: v for k, v in init_state.items()
        if k in own and own[k].shape == v.shape
    }
    own.update(compatible)
    model.load_state_dict(own)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
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

        out = model(toks, target_weights=target_weights)
        utility = None
        utility_weight = 0.0
        total_loss = out["loss"]

        if kind == "set-utility-3step" and step >= args.utility_start:
            utility = set_conditioned_utility_loss(
                model,
                toks,
                answer_mask[:, 1:],
                temperature=args.utility_temperature,
                min_gain_std=args.utility_min_std,
            )
            progress = (step - args.utility_start + 1) / max(1, args.utility_warmup)
            utility_weight = args.utility_weight * min(1.0, progress)
            total_loss = total_loss + utility_weight * utility["loss"]

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
            row["reasoning_steps"] = float(model.reasoning_steps)
            row["set_utility_weight"] = float(utility_weight)
            if utility is not None:
                row.update(
                    {
                        f"set_utility_{k}": float(v.detach())
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="casm-r-output")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=289)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--answer-weight", type=float, default=12.0)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=800)
    p.add_argument("--utility-start", type=int, default=400)
    p.add_argument("--utility-warmup", type=int, default=400)
    p.add_argument("--utility-weight", type=float, default=0.10)
    p.add_argument("--utility-temperature", type=float, default=0.7)
    p.add_argument("--utility-min-std", type=float, default=0.01)
    p.add_argument("--tiny", action="store_true")
    p.add_argument(
        "--models",
        default="qk-1step,qk-3step,set-utility-3step",
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
            use_compression_score=False,
            compression_loss_weight=0.0,
            compression_predictor_loss_weight=0.0,
        )

    probe = CASMRecurrent(cfg, reasoning_steps=1)
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
            {
                "state_dict": model.state_dict(),
                "config": asdict(model.cfg),
                "reasoning_steps": model.reasoning_steps,
                "kind": kind,
            },
            out_dir / f"{kind}.pt",
        )
        (out_dir / f"{kind}-history.json").write_text(
            json.dumps(history, indent=2)
        )
        summary.append(
            {
                "model": kind,
                "parameters": model.parameter_count(),
                "reasoning_steps": model.reasoning_steps,
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
