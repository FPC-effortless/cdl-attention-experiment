from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List

import torch

from .data import BOS, EOS, PAD, VOCAB_SIZE
from .model import CASMConfig
from .process_data import ProcessExample, graph_process, state_process
from .recurrent_model import CASMRecurrent


def build_core(cfg: CASMConfig) -> CASMRecurrent:
    cfg = replace(
        cfg,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    return CASMRecurrent(cfg, reasoning_steps=3)


def task_batch(
    fn: Callable[[random.Random, bool], ProcessExample],
    batch_size: int,
    seq_len: int,
    seed: int,
    hard: bool,
):
    rng = random.Random(seed)
    rows: List[List[int]] = []
    masks: List[List[bool]] = []
    marker = b"answer "
    for _ in range(batch_size):
        for _attempt in range(2000):
            pex = fn(rng, hard)
            body = pex.example.text.encode("utf-8", errors="replace")
            if len(body) + 2 <= seq_len:
                break
        else:
            raise RuntimeError("could not sample episode fitting seq_len")
        at = body.rfind(marker)
        if at < 0:
            raise ValueError("missing answer marker")
        answer_start = 1 + at + len(marker)
        answer_len = len(pex.example.answer.encode("utf-8", errors="replace"))
        row = [BOS] + list(body) + [EOS]
        mask = [False] * len(row)
        for j in range(answer_start, min(answer_start + answer_len, len(row) - 1)):
            mask[j] = True
        pad = seq_len - len(row)
        row.extend([PAD] * pad)
        mask.extend([False] * pad)
        rows.append(row)
        masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def train_task(name: str, fn, args, cfg, init_state: Dict[str, torch.Tensor]):
    model = build_core(cfg)
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    history = []
    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(1.0, step / max(1, args.hard_ramp_steps))
        hard = random.Random(args.seed + step * 65537 + (1 if name == "graph" else 0)).random() < hard_prob
        toks, answer_mask = task_batch(fn, args.batch_size, args.seq_len, args.seed + step * 104729, hard)
        weights = 1.0 + (args.answer_weight - 1.0) * answer_mask[:, 1:].float()
        out = model(toks, target_weights=weights)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "lm_loss": float(out["lm_loss"].detach()),
                "router_entropy": float(out["router_entropy"].detach()),
                "memory_gate_mean": float(out["memory_gate_mean"].detach()),
            }
            history.append(row)
            print(name, row, flush=True)
    return model, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="capacity-floor-output")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=385)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261301)
    p.add_argument("--answer-weight", type=float, default=12.0)
    p.add_argument("--hard-max-prob", type=float, default=0.7)
    p.add_argument("--hard-ramp-steps", type=int, default=400)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    cfg = replace(
        CASMConfig(vocab_size=VOCAB_SIZE),
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
    init_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("state", state_process), ("graph", graph_process)):
        model, history = train_task(name, fn, args, cfg, init_state)
        torch.save({
            "state_dict": model.state_dict(),
            "config": asdict(model.cfg),
            "reasoning_steps": 3,
            "kind": f"capacity-{name}",
        }, out / f"{name}-only.pt")
        (out / f"{name}-history.json").write_text(json.dumps(history, indent=2))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()
