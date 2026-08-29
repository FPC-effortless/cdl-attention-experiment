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
import torch.nn.functional as F

from .data import PAD, VOCAB_SIZE
from .model import CASMConfig
from .recurrent_model import CASMRecurrent
from .scoped_data import make_episode_batch


def build_model(kind: str, cfg: CASMConfig) -> CASMRecurrent:
    common = replace(
        cfg,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    if kind == "scoped-qk-1step":
        return CASMRecurrent(common, reasoning_steps=1)
    if kind == "scoped-qk-3step":
        return CASMRecurrent(common, reasoning_steps=3)
    raise ValueError(kind)


def token_accuracy(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    target = tokens[:, 1:]
    pred = logits.argmax(dim=-1)
    mask = target != PAD
    return float((pred[mask] == target[mask]).float().mean()) if mask.any() else 0.0


@torch.inference_mode()
def evaluate_scoped(
    model: CASMRecurrent,
    cfg: CASMConfig,
    seed: int,
    batches: int,
    batch_size: int,
    seq_len: int,
    hard: bool,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    answer_nlls: List[float] = []
    for i in range(batches):
        toks, _, answer_mask = make_episode_batch(
            batch_size,
            seq_len,
            seed + i * 7919,
            hard=hard,
            return_answer_mask=True,
        )
        out = model(toks, return_aux=False)
        losses.append(float(out["lm_loss"]))
        accs.append(token_accuracy(out["logits"], toks))
        target = toks[:, 1:]
        mask = answer_mask[:, 1:] & (target != PAD)
        if mask.any():
            tok_nll = F.cross_entropy(
                out["logits"].transpose(1, 2),
                target,
                ignore_index=PAD,
                reduction="none",
            )
            answer_nlls.append(float(tok_nll[mask].mean()))
    return {
        "loss": sum(losses) / len(losses),
        "token_accuracy": sum(accs) / len(accs),
        "answer_nll": sum(answer_nlls) / max(1, len(answer_nlls)),
    }


def train_one(kind: str, args, cfg: CASMConfig, init_state: Dict[str, torch.Tensor]):
    model = build_model(kind, cfg)
    own = model.state_dict()
    compatible = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
    own.update(compatible)
    model.load_state_dict(own)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(1.0, step / max(1, args.hard_ramp_steps))
        use_hard = random.Random(args.seed + step * 65537).random() < hard_prob
        toks, _, answer_mask = make_episode_batch(
            args.batch_size,
            args.seq_len,
            args.seed + step * 104729,
            hard=use_hard,
            return_answer_mask=True,
        )
        target_weights = 1.0 + (args.answer_weight - 1.0) * answer_mask[:, 1:].float()
        out = model(toks, target_weights=target_weights)

        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {k: float(v.detach()) for k, v in out.items() if k != "logits"}
            row["step"] = step
            row["reasoning_steps"] = float(model.reasoning_steps)
            history.append(row)
            print(kind, row, flush=True)

    elapsed = time.perf_counter() - started
    easy = evaluate_scoped(
        model, cfg, args.seed + 9_000_000, args.eval_batches,
        args.batch_size, args.seq_len, False,
    )
    hard = evaluate_scoped(
        model, cfg, args.seed + 19_000_000, args.eval_batches,
        args.batch_size, args.seq_len, True,
    )
    return model, history, easy, hard, elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="casm-r-scoped-output")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=289)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--answer-weight", type=float, default=12.0)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=800)
    p.add_argument("--tiny", action="store_true")
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
    for kind in ("scoped-qk-1step", "scoped-qk-3step"):
        model, history, easy, hard, elapsed = train_one(kind, args, cfg, init)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(model.cfg),
                "reasoning_steps": model.reasoning_steps,
                "kind": kind,
                "training_scope": "one-independent-task-per-row",
            },
            out_dir / f"{kind}.pt",
        )
        (out_dir / f"{kind}-history.json").write_text(json.dumps(history, indent=2))
        summary.append({
            "model": kind,
            "parameters": model.parameter_count(),
            "reasoning_steps": model.reasoning_steps,
            "seconds": elapsed,
            **{f"easy_{k}": v for k, v in easy.items()},
            **{f"hard_{k}": v for k, v in hard.items()},
        })

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
