"""COG-COMM-001 follow-up: frozen A endpoint on original EASY TRAIN only.

Do not launch directly as an experiment. Use a fresh independently reviewed
Forge/TrainLab closure with exactly the original inputs and bounded CPU run.
Never substitute regenerated data or refitted weights. This is an in-sample
diagnostic, not CAL/DEV/TEST or an alteration of the official gate.
"""
import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
import math
from pathlib import Path
import re
import sys

SEEDS = (171, 271, 371)
CHECKPOINT_KEYS = {
    "format", "plan", "tokenizer_words", "vocab_size", "train_sha256",
    "vocab_sha256", "schedule_sha256", "updates", "state_dict",
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def exact_file(path, expected):
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "exact SHA-256 required")
    with path.open("rb") as f:
        require(file_digest(f, "sha256").hexdigest() == expected, "input digest differed")

def outcome(left, right, a, b):
    require(len(left) == len(right) and 0 <= a < len(left)
            and 0 <= b < len(left) and a != b, "invalid pair")
    require(all(math.isfinite(v) for row in (left, right) for v in row), "nonfinite logits")
    h = tuple(row[target] > max(row[:target] + row[target+1:])
              for row, target in ((left, a), (right, b)))
    margins = (left[a] - left[b], right[b] - right[a])
    o = ("tie" if 0 in margins else "correct" if min(margins) > 0
         else "reversed" if max(margins) < 0 else "same_color")
    require(not all(h) or o == "correct", "strict rank containment violated")
    return h, o

def evaluate(args):
    # Hashes are independently recovered/admitted, not calculated to bless
    # whichever files happen to be beside this worker.
    for k in ("train", "vocab", "checkpoint"):
        exact_file(getattr(args, k), getattr(args, k+"_sha256"))
    require(not args.out.exists(), "output already exists")
    import torch
    from .fit import Plan, batch_inputs, encode, metadata, order_for, read_train, read_vocab
    from .model import build_decoder

    require(sys.version_info[:2] == (3, 12) and torch.__version__ == "2.10.0+cpu",
            "wrong frozen CPU runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = Plan("A", "learned", args.seed)
    families = read_train(args.train)
    require(len(families) == 320, "wrong original EASY TRAIN count")
    tok = read_vocab(args.vocab, families)
    require(tok.size == 280, "wrong EASY vocabulary size")
    rows = encode(families, tok, plan.max_tokens)
    order = order_for(len(rows), plan)
    pre = metadata(rows, plan, order, tok)
    require(pre["unique_families_observed"] == len(rows), "not all rows were seen during fit")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(ck, dict) and ck.keys() == CHECKPOINT_KEYS and
            ck["format"] == 1 and ck["plan"] == asdict(plan) and ck["updates"] == plan.steps
            and ck["tokenizer_words"] == list(tok.words) and ck["vocab_size"] == tok.size
            and ck["train_sha256"] == args.train_sha256 and ck["vocab_sha256"] == args.vocab_sha256
            and ck["schedule_sha256"] == pre["schedule_sha256"], "checkpoint lineage mismatch")
    m = build_decoder(tok.size, "A", width=plan.width, layers=plan.layers, block=plan.block,
                      retained_blocks=plan.retained_blocks, seed=plan.seed).eval()
    require(all(t.dtype == torch.float32 and t.device.type == "cpu" and
                bool(torch.isfinite(t).all()) for t in ck["state_dict"].values()), "weight format drift")
    m.load_state_dict(ck["state_dict"], strict=True)
    by_color = {str(tok.index[color]): [0, 0] for color in ("blue","gold","green","orange","pink","purple","red","white")}
    orientation = dict.fromkeys(("tie", "correct", "reversed", "same_color"), 0)
    worlds = pairs = calls = tokens = 0
    # Use the first-epoch evaluation geometry to match the original batches.
    first_epoch = order[:len(rows)//(plan.worlds_per_step//2)]
    require(sorted(i for batch in first_epoch for i in batch) == list(range(len(rows))),
            "first-epoch coverage did not partition TRAIN")
    with torch.inference_mode():
        for batch in first_epoch:
            x, lengths, y, _ = batch_inputs(rows, batch)
            logits, _ = m(x, state=None)
            require(logits.shape == (*x.shape, tok.size), "unexpected logit shape")
            v = logits[torch.arange(len(x)), lengths-1].tolist()
            tokens += int(lengths.sum())
            calls += 1
            for j, (a,b) in enumerate(y.tolist()):
                h, o = outcome(v[2*j], v[2*j+1], a,b)
                worlds += sum(h)
                pairs += all(h)
                orientation[o] += 1
                for target, hit in zip((a,b), h):
                    by_color[str(target)][0] += hit
                    by_color[str(target)][1] += 1
    require(sum(orientation.values()) == len(rows) and calls == 80 and
            sum(n for h,n in by_color.values()) == 640, "incomplete scorer coverage")
    result = dict(scope="EASY original TRAIN fixed endpoint only", seed=args.seed,
                  checkpoint_sha256=args.checkpoint_sha256, train_sha256=args.train_sha256,
                  vocab_sha256=args.vocab_sha256, model_calls=calls, optimizer_steps=0,
                  prompt_tokens=tokens, worlds_correct=worlds, worlds_total=640,
                  exact_pairs=pairs, pairs_total=320, sibling_orientation=orientation,
                  full_vocabulary_by_color=by_color)
    with args.out.open("x", encoding="utf8") as fp:
        json.dump(result, fp, allow_nan=False, sort_keys=True)
        fp.write("\n")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("train", "vocab", "checkpoint"):
        ap.add_argument("--"+k, type=Path, required=True)
        ap.add_argument("--"+k+"-sha256", required=True)
    ap.add_argument("--seed", choices=SEEDS, type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    evaluate(ap.parse_args())

if __name__ == "__main__":
    main()
