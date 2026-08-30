from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .data import VOCAB_SIZE, gzip_teacher_distributions, make_batch
from .model import CASM, CASMConfig
from .selective import SelectiveCASM
from .train import evaluate, train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="selective-replay")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=193)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--compression-warmup", type=int, default=125)
    p.add_argument("--gzip-teacher-decay", type=int, default=650)
    p.add_argument("--gzip-temperature", type=float, default=1.0)
    p.add_argument("--gzip-teacher-floor", type=float, default=0.35)
    p.add_argument("--answer-weight", type=float, default=8.0)
    p.add_argument("--hard-max-prob", type=float, default=0.5)
    p.add_argument("--hard-ramp-steps", type=int, default=500)
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
    )
    probe = CASM(cfg)
    init = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact v0.1 compression-trained Q/K control using the same base initialization.
    control, history, easy, hard, elapsed = train_one("compression-qk", args, cfg, init_state=init)
    torch.save({"state_dict": control.state_dict(), "config": asdict(control.cfg)}, out_dir / "compression-qk.pt")
    (out_dir / "compression-qk-history.json").write_text(json.dumps(history, indent=2))

    candidate = SelectiveCASM(cfg)
    own = candidate.state_dict()
    compatible = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape}
    own.update(compatible)
    candidate.load_state_dict(own)
    opt = torch.optim.AdamW(candidate.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    final_compression_weight = candidate.cfg.compression_loss_weight
    selective_history = []
    candidate.train()
    for step in range(1, args.steps + 1):
        hard_prob = args.hard_max_prob * min(1.0, step / max(1, args.hard_ramp_steps))
        use_hard = random.Random(args.seed + step * 65537).random() < hard_prob
        toks, _, answer_mask = make_batch(
            args.batch_size, args.seq_len, args.seed + step * 104729,
            hard=use_hard, return_answer_mask=True,
        )
        target_weights = torch.ones_like(toks[:, 1:], dtype=torch.float32) + (
            args.answer_weight - 1.0
        ) * answer_mask[:, 1:].float()
        candidate.cfg.compression_loss_weight = final_compression_weight * min(
            1.0, step / max(1, args.compression_warmup)
        )
        external = gzip_teacher_distributions(
            toks, candidate.cfg.chunk_size, candidate.cfg.memory_slots,
            candidate.cfg.state_slots, args.gzip_temperature,
        )
        decay = max(0.0, 1.0 - step / max(1, args.gzip_teacher_decay))
        alpha = args.gzip_teacher_floor + (1.0 - args.gzip_teacher_floor) * decay
        result = candidate(
            toks, external_teacher=external, teacher_alpha=alpha,
            target_weights=target_weights,
        )
        opt.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {k: float(v.detach()) for k, v in result.items() if k != "logits"}
            row["step"] = step
            selective_history.append(row)
            print("selective", row, flush=True)

    selective_easy = evaluate(candidate, cfg, args.seed + 9_000_000, args.eval_batches, args.batch_size, False)
    selective_hard = evaluate(candidate, cfg, args.seed + 19_000_000, args.eval_batches, args.batch_size, True)
    torch.save(
        {"state_dict": candidate.state_dict(), "config": asdict(candidate.cfg), "kind": "selective"},
        out_dir / "selective.pt",
    )
    (out_dir / "selective-history.json").write_text(json.dumps(selective_history, indent=2))
    (out_dir / "smoke-summary.json").write_text(json.dumps({
        "status": "smoke-level replay",
        "steps": args.steps,
        "seed": args.seed,
        "control": {"easy": easy, "hard": hard, "seconds": elapsed},
        "selective": {"easy": selective_easy, "hard": selective_hard},
    }, indent=2))


if __name__ == "__main__":
    main()
