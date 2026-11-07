"""Reproducible CPU engineering pilot, explicitly separate from confirmation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .data import make_example, batch
from .model import Config, RelationalLM
from .train import counterfactual_step


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


@torch.no_grad()
def evaluate(model, examples, memory="normal"):
    model.eval()
    x, y, mask = batch(examples)
    output = model(x, memory=memory)
    loss = F.cross_entropy(output.logits.flatten(0, 1), y.flatten(),
                           ignore_index=0, reduction="none").reshape_as(y)
    predictions = output.logits.argmax(-1)
    exact = ((predictions == y) | ~mask).all(-1)
    return dict(nll=float(loss.sum() / (y != 0).sum()),
                answer_nll=float(loss[mask].mean()),
                teacher_forced_answer_exact=float(exact.float().mean()),
                # Structural metadata is diagnostic only, not unique learned semantics.
                final_query_fact_end_hits=sum(
                    len(set(output.operands[-1][i, e.answer_start - 1].tolist()) & set(e.fact_ends))
                    for i, e in enumerate(examples)),
                examples=len(examples), metrics=output.metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-examples", type=int, default=8)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--variants", nargs="+", default=["coupled", "independent", "no_reference"])
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_examples, args.width) < 1:
        p.error("positive steps, batch size, evaluation size and width required")
    if args.resume and (len(args.seeds) != 1 or len(args.variants) != 1):
        p.error("resume requires exactly one seed and variant")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = dict(status="ENGINEERING_PILOT_NOT_CONFIRMATORY", source_sha256=hashes,
                    torch=torch.__version__, python=platform.python_version(),
                    device="cpu", threads=1, settings={k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()}, results=[])
    # This manifest is written BEFORE any fit; no adaptive hyperparameter selection.
    manifest_path = args.output / "pilot.json"
    if manifest_path.exists():
        p.error("output already has a pilot manifest; choose a fresh directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            config = Config(width=args.width, variant=variant)
            model = RelationalLM(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
            trace, first_step = [], 0
            if args.resume:
                saved = torch.load(args.resume, map_location="cpu", weights_only=True)
                if saved["config"] != asdict(config) or saved["seed"] != seed or saved["source_sha256"] != hashes:
                    raise ValueError("resume config, seed or source mismatch")
                if saved["batch_size"] != args.batch_size:
                    raise ValueError("resume batch size mismatch")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["rng"])
                first_step, trace = saved["step"], saved["trace"]
                if args.steps <= first_step:
                    raise ValueError("resume steps must exceed checkpoint step")
            before = evaluate(model, [make_example(700_000 + i, "dev") for i in range(args.eval_examples)])
            started = time.perf_counter()
            for step in range(first_step, args.steps):
                examples = [make_example(seed * 1_000_000 + step * args.batch_size + i)
                            for i in range(args.batch_size)]
                x, y, _ = batch(examples)
                trace.append(counterfactual_step(model, optimizer, x, y, seed=seed * 10_000 + step))
            train_seconds = time.perf_counter() - started
            checkpoint = args.output / f"{variant}-{seed}.pt"
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            rng=torch.get_rng_state(), step=args.steps, config=asdict(config),
                            seed=seed, batch_size=args.batch_size, trace=trace,
                            source_sha256=hashes), checkpoint)
            # DEV only: final confirmation seeds and TEST remain unopened in this pilot.
            conditions = {}
            eval_start = time.perf_counter()
            for condition in ("iid", "names", "composition", "paraphrase", "updates", "older"):
                examples = [make_example(700_000 + i, "dev", condition) for i in range(args.eval_examples)]
                conditions[condition] = evaluate(model, examples)
            examples = [make_example(700_000 + i, "dev") for i in range(args.eval_examples)]
            controls = {m: evaluate(model, examples, m) for m in ("disabled", "shuffled")}
            result = dict(seed=seed, variant=variant, config=asdict(config),
                          parameters=sum(p.numel() for p in model.parameters()),
                          initial_dev=before, dev=conditions, controls=controls,
                          train_seconds_this_invocation=train_seconds,
                          evaluation_seconds=time.perf_counter() - eval_start,
                          training_forward_passes=sum(t["forward_passes"] for t in trace),
                          training_cost={k: sum(t["cost"][k] for t in trace) for k in trace[0]["cost"]},
                          nonzero_intervention_effects=sum(t["advantage"] != 0 for t in trace),
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          trace=trace)
            manifest["results"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({k: result[k] for k in ("seed", "variant", "parameters",
                "train_seconds_this_invocation", "nonzero_intervention_effects")}), flush=True)


if __name__ == "__main__":
    main()
