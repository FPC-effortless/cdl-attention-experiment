"""Independent data/evaluation/checkpoint tests; no fit to corpus targets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from data import CharTokenizer, load_split
from model import COGTinyLM, ModelConfig
from sample import generate, load_model
from train import (atomic_checkpoint, capture_rng, cpu_weights, evaluate,
                   get_batch, learning_rate, restore_rng)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19317)
    random.seed(19317)
    here = Path(__file__).resolve().parent
    begun = time.perf_counter()
    checks = []

    def record(name, passed, **evidence):
        checks.append({"name": name, "passed": bool(passed), **evidence})

    data = here / "data"
    meta = json.loads((data / "dataset.json").read_text())
    raw = (data / "input.txt").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    record("pinned_source_identity", digest == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed" == meta["input_sha256"],
           bytes=len(raw), sha256=digest)
    text = raw.decode("utf-8")
    tokenizer = CharTokenizer.load(data / "tokenizer.json")
    train_edge = meta["splits"]["train"]["end_character_exclusive"]
    record("vocabulary_is_training_characters_only", tokenizer.chars == sorted(set(text[:train_edge])),
           vocab_size=len(tokenizer.chars), train_characters=train_edge)
    expected_start = 0
    splits_exact = True
    split_targets = {}
    for split in ("train", "val", "test"):
        info = meta["splits"][split]
        a, b = info["start_character_inclusive"], info["end_character_exclusive"]
        ids = load_split(data, split)
        decoded = tokenizer.decode(ids.tolist())
        splits_exact &= a == expected_start and decoded == text[a:b]
        split_targets[split] = len(ids) - 1
        expected_start = b
    record("all_split_bytes_and_contiguous_boundaries_match", splits_exact and expected_start == len(text),
           next_character_targets=split_targets)
    try:
        tokenizer.encode("\N{SNOWMAN}")
        rejected = False
    except ValueError:
        rejected = True
    record("unseen_prompt_character_is_explicitly_rejected", rejected)

    # Each token here is its own position, so an offset bug is observable.
    positions = torch.arange(37, dtype=torch.long)
    sampler = torch.Generator().manual_seed(444)
    x, y = get_batch(positions, 4096, 13, sampler, torch.device("cpu"))
    valid = bool((x[:, 0] >= 0).all() and (x[:, -1] <= 35).all() and (y[:, -1] <= 36).all()
                 and torch.equal(y, x + 1))
    record("sampler_inputs_and_targets_stay_inside_split", valid,
           observed_min_start=int(x[:, 0].min()), observed_max_start=int(x[:, 0].max()),
           allowed_max_start=23, sampled_windows=4096)

    class PositionPredictor(torch.nn.Module):
        def forward(self, tokens, state=None):
            logits = F.one_hot((tokens + 1) % 5, 5).float() * 3.0
            return logits, None

    test_tokens = torch.arange(24, dtype=torch.long) % 5
    scorer = PositionPredictor().train()
    assessed = evaluate(scorer, test_tokens, 7, 3, torch.device("cpu"))
    independent_logits = F.one_hot((test_tokens[:-1] + 1) % 5, 5).float() * 3.0
    expected_nll = float(F.cross_entropy(independent_logits, test_tokens[1:], reduction="none").double().mean())
    record("evaluation_covers_each_target_once_including_partial_window",
           assessed["targets"] == 23 and assessed["next_character_accuracy"] == 1.0
           and abs(assessed["mean_nll"] - expected_nll) < 1e-7 and scorer.training,
           result=assessed, independently_computed_nll=expected_nll)

    # Two exact next-step continuations from a saved state, using a small
    # instance and constructed integers. No Shakespeare target enters updates.
    cfg = ModelConfig(vocab_size=5, d_model=12, n_layers=2, memory_dim=4,
                      n_entities=2, n_relations=2, dropout=0.2)
    model = COGTinyLM(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(817)
    artificial = torch.tensor(([0, 1, 4, 2, 3, 1, 3, 0, 4] * 10), dtype=torch.long)
    schedule = {"warmup_steps": 1, "max_steps": 5, "learning_rate": 3e-4, "min_lr_ratio": 0.03}

    def update(net, optimizer, rng, step):
        bx, by = get_batch(artificial, 2, 11, rng, torch.device("cpu"))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, schedule)
        optimizer.zero_grad(set_to_none=True)
        scores, _ = net(bx)
        loss = F.cross_entropy(scores.flatten(0, 1), by.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        return bx, by, float(loss.detach())

    update(model, opt, generator, 0)
    with tempfile.TemporaryDirectory(prefix="cog_review_") as folder:
        path = Path(folder) / "checkpoint.pt"
        payload = {"model": cpu_weights(model), "model_config": asdict(cfg),
                   "optimizer": opt.state_dict(), "step": 1,
                   "rng": capture_rng(generator),
                   "tokenizer": {"chars": list("abcde")}}
        atomic_checkpoint(path, payload)
        batch_a, targets_a, loss_a = update(model, opt, generator, 1)
        weights_a = {k: v.clone() for k, v in model.state_dict().items()}
        next_python_a = random.random()
        next_torch_a = torch.rand(9)

        restored = torch.load(path, map_location="cpu", weights_only=True)
        other = COGTinyLM(ModelConfig(**restored["model_config"])).train()
        other.load_state_dict(restored["model"])
        other_opt = torch.optim.AdamW(other.parameters(), lr=3e-4, betas=(0.9, 0.95))
        other_opt.load_state_dict(restored["optimizer"])
        other_generator = torch.Generator()
        restore_rng(restored["rng"], other_generator)
        batch_b, targets_b, loss_b = update(other, other_opt, other_generator, restored["step"])
        next_python_b = random.random()
        next_torch_b = torch.rand(9)
        max_delta = max(float((weights_a[k] - v).abs().max()) for k, v in other.state_dict().items())
        record("checkpoint_restore_reproduces_next_optimizer_update",
               torch.equal(batch_a, batch_b) and torch.equal(targets_a, targets_b)
               and loss_a == loss_b and max_delta == 0.0,
               loss_before_restore=loss_a, loss_after_restore=loss_b, parameter_max_abs_error=max_delta,
               fixture="small same-architecture model; constructed integer sequence; two steps per path")
        record("python_torch_and_sampler_rng_recovered",
               next_python_a == next_python_b and torch.equal(next_torch_a, next_torch_b)
               and torch.equal(generator.get_state(), other_generator.get_state()))

        loaded, char_tokens, metadata = load_model(path, threads=2)
        text_a, _ = generate(loaded, char_tokens, "ab", 31, 0.8, 0, 714)
        text_b, _ = generate(loaded, char_tokens, "ab", 31, 0.8, 0, 714)
        record("sampling_entrypoint_reloads_and_repeats_seeded_generation",
               text_a == text_b and len(text_a) == 33 and text_a.startswith("ab"),
               characters=len(text_a), generated_characters=31, seed=714)

    result = {
        "scope": "independent pipeline checks; constructed optimizer updates only, no corpus fit",
        "sources": {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                    for name in ("model.py", "data.py", "train.py", "sample.py", "verify_pipeline.py")},
        "torch_version": str(torch.__version__), "device": "cpu", "threads": 2,
        "elapsed_seconds": time.perf_counter() - begun,
        "passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks,
    }
    (here / "pipeline_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "total", "elapsed_seconds")}, indent=2))
    for c in checks:
        if not c["passed"]:
            print(json.dumps(c, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
