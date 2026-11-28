"""Fine-tune matched pilot checkpoints on causal evidence and query revisions."""
import argparse
import hashlib
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cog-balance-001"))
from corpus import load_corpus
from panels import evaluate
from profiler import profile_model
from scale import build_model
from confirm_repair import evaluate_confirmation
from repair_objective import repair_batch, repair_loss, REGIMES


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def assess(model, corpus):
    model.eval()
    losses = []
    rng = np.random.default_rng(781299)
    for _ in range(16):
        xy = torch.from_numpy(corpus.batch("dev", rng, 8, 128))
        logits = model(xy[:, :-1])
        losses.append(F.cross_entropy(logits.flatten(0, 1), xy[:, 1:].flatten()).item())
    result = {"dev_bpb":np.mean(losses).item() / math.log(2),
              "panels":evaluate(model, 610003, worlds=64, batch_size=12),
              "confirmation":evaluate_confirmation(model, worlds=64, batch_size=8),
              "profile":profile_model(model, xy)}
    return result


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    corpus = load_corpus(args.corpus)
    state = torch.load(args.parent, weights_only=True)
    if args.arm == "R":
        from repair_model import build_repair_model
        model = build_repair_model(args.seed, parent_state=state)
    else:
        model = build_model(args.arm, args.seed)
        model.load_state_dict(state)
    args.out.mkdir(exist_ok=True, parents=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(args.seed + 311903)
    digest = hashlib.sha256()
    replay_targets = prompt_bytes = 0
    model.train()
    start = time.process_time()
    for step in range(args.steps):
        b = repair_batch(args.seed, step, batch_size=8, balanced=args.data == "balanced")
        prompt_bytes += b.ids.numel()
        for tensor in (b.ids, b.final_targets, b.trace_targets):
            digest.update(tensor.numpy().tobytes())
        opt.zero_grad(set_to_none=True)
        loss = repair_loss(model(b.ids), b, args.regime)
        if not torch.isfinite(loss):
            raise ArithmeticError("non-finite cognitive loss")
        loss.backward()
        if step % 4 == 0:
            xy = torch.from_numpy(corpus.batch("train", rng, 4, 128))
            replay_targets += xy[:, 1:].numel()
            digest.update(xy.numpy().tobytes())
            logits = model(xy[:, :-1])
            lang = F.cross_entropy(logits.flatten(0,1), xy[:,1:].flatten())
            lang.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    cpu = time.process_time() - start
    torch.save(model.state_dict(), args.out / "weights.pt")
    result = {"arm":args.arm, "seed":args.seed, "data":args.data, "regime":args.regime,
              "parameters":sum(p.numel() for p in model.parameters()), "steps":args.steps,
              "question_streams":args.steps * 8, "question_bytes":prompt_bytes, "real_text_replay_targets":replay_targets,
              "training_sha256":digest.hexdigest(), "parent_sha256":sha(args.parent),
              "trained_weights_sha256":sha(args.out/"weights.pt"), "training_cpu_seconds":cpu,
              "assessment":assess(model, corpus)}
    if args.arm == "R":
        model.disable_read = True
        result["read_disabled"] = {"panels":evaluate(model,610003,worlds=64,batch_size=12),
                                   "confirmation":evaluate_confirmation(model,worlds=64,batch_size=8)}
    if result["assessment"]["profile"]["batch_vs_stream_max_abs_logit_error"] > 0.002:
        raise ArithmeticError("trained streaming check failed")
    tmp=args.out/"metrics.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False, indent=2)+"\n")
    tmp.replace(args.out/"metrics.json")
    print(args.arm, args.regime, args.seed, result["assessment"]["dev_bpb"], flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=["C","R","H","GQA","GRU"],required=True)
    p.add_argument("--regime",choices=REGIMES,required=True)
    p.add_argument("--data",choices=["balanced","original"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=256)
    for key in ("parent","out","corpus"):
        p.add_argument("--"+key, type=Path, required=True)
    args=p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    run(args)
