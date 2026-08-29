from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import BOS, EOS, TASKS, Example
from .model import CASM, CASMConfig


def load_model(path: str) -> CASM:
    ckpt = torch.load(path, map_location="cpu")
    cfg = CASMConfig(**ckpt["config"])
    model = CASM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def prefix_answer(ex: Example):
    prefix, answer = ex.text.rsplit("answer ", 1)
    return prefix + "answer ", answer


@torch.inference_mode()
def score_example(model: CASM, ex: Example) -> Dict[str, float]:
    prefix, answer = prefix_answer(ex)
    p = list(prefix.encode("utf-8")); a = list(answer.encode("utf-8"))
    tokens = torch.tensor([[BOS] + p + a + [EOS]], dtype=torch.long)
    out = model(tokens, return_aux=False)
    logits = out["logits"][0]
    ans_start = 1 + len(p)
    positions = torch.arange(ans_start - 1, ans_start - 1 + len(a))
    selected = logits[positions]; target = torch.tensor(a, dtype=torch.long)
    nll = float(F.cross_entropy(selected, target)) if a else 0.0
    pred = selected.argmax(dim=-1)
    return {
        "answer_nll": nll,
        "answer_byte_acc": float((pred == target).float().mean()) if a else 1.0,
        "answer_exact_tf": float(bool(torch.equal(pred, target))),
        "context_bytes": len(p),
    }


def eval_model(model: CASM, seed: int, n_per_task: int, hard: bool) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed); rows: Dict[str, List[Dict[str, float]]] = {}
    for fn in TASKS:
        rows[fn.__name__] = [score_example(model, fn(rng, hard=hard)) for _ in range(n_per_task)]
    out = {name: {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]} for name, vals in rows.items()}
    flat = [v for vals in rows.values() for v in vals]
    out["overall"] = {k: sum(v[k] for v in flat) / len(flat) for k in flat[0]}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("checkpoints", nargs="+"); p.add_argument("--n-per-task", type=int, default=30); p.add_argument("--seed", type=int, default=777)
    args = p.parse_args(); results = {}
    for cp in args.checkpoints:
        model = load_model(cp)
        results[Path(cp).stem] = {"easy": eval_model(model, args.seed, args.n_per_task, False), "hard": eval_model(model, args.seed + 100000, args.n_per_task, True)}
        print(Path(cp).stem, json.dumps(results[Path(cp).stem]["hard"]["overall"], indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
