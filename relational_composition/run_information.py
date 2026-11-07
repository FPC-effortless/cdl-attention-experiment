"""Fixed LRC-002 diagnostic experiment; all fits close before held-out evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from .data import batch
from .information import ARMS, example, forward_options, measure
from .model import Config, RelationalLM
from .run import source_hashes
from .train import counterfactual_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds, steps, batch_size = (201, 202, 203), 400, 4
    config = Config(width=24, layers=2)
    hashes = source_hashes()
    protocol = Path(__file__).with_name("EXPERIMENT_LRC002.md")
    manifest = dict(status="PREREGISTERED_DIAGNOSTIC_NOT_ARCHITECTURE_QUALIFICATION",
        source_sha256=hashes, protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=str(torch.__version__), python=platform.python_version(), config=asdict(config),
        seeds=seeds, steps=steps, batch_size=batch_size, learning_rate=0.002,
        random_answer_entropy_bits=math.log2(20), random_answer_chance=0.05,
        phase="training", results=[])
    path = args.output / "manifest.json"
    def save():
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    for seed in seeds:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = RelationalLM(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
            trace = []
            started = time.perf_counter()
            for step in range(steps):
                examples = [example(20_000_000 + seed * 100_000 + step * batch_size + i,
                            random_answer=arm == "oracle_random_answer") for i in range(batch_size)]
                x, y, mask = batch(examples)
                options = forward_options(arm, examples, x, config.layers)
                trace.append(counterfactual_step(model, optimizer, x, y,
                    seed=seed * 100_000 + step, forward_kwargs=options,
                    loss_mask=mask if "answer" in arm else None))
            elapsed = time.perf_counter() - started
            checkpoint = args.output / f"{arm}-{seed}.pt"
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(), trace=trace, config=asdict(config), seed=seed,
                arm=arm, steps=steps, source_sha256=hashes), checkpoint)
            result = dict(seed=seed, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                train_seconds=elapsed, first_objective=trace[0]["nll"], last_objective=trace[-1]["nll"],
                forward_passes=sum(t["forward_passes"] for t in trace),
                train_cost={k: sum(t["cost"][k] for t in trace) for k in trace[0]["cost"]},
                nonzero_credit=sum(t["advantage"] != 0 for t in trace),
                checkpoint=checkpoint.name, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            manifest["results"].append(result)
            save()
            print(json.dumps({k: result[k] for k in ("seed", "arm", "train_seconds", "nonzero_credit")}), flush=True)
    # No held-out example is constructed above this line.
    manifest["phase"] = "held_out_evaluation"
    save()
    for result in manifest["results"]:
        saved = torch.load(args.output / result["checkpoint"], weights_only=True)
        model = RelationalLM(config)
        model.load_state_dict(saved["model"])
        started = time.perf_counter()
        result["held_out"] = {}
        for condition in ("iid", "composition", "names"):
            # Noise-label population stays IID: its specified uniform-20 bound
            # does not share the unseen-name alphabet diagnostic.
            if result["arm"] == "oracle_random_answer" and condition != "iid":
                continue
            examples = [example(90_000_000 + i, condition=condition,
                        random_answer=result["arm"] == "oracle_random_answer") for i in range(64)]
            result["held_out"][condition] = measure(model, examples, result["arm"])
        result["evaluation_seconds"] = time.perf_counter() - started
        save()
        print(json.dumps(dict(seed=result["seed"], arm=result["arm"],
                              accuracy=result["held_out"]["iid"]["generated_accuracy"])), flush=True)
    oracle = [r for r in manifest["results"] if r["arm"] == "oracle_answer"]
    noise = [r for r in manifest["results"] if r["arm"] == "oracle_random_answer"]
    manifest["gates"] = dict(
        oracle_decoder_competence=all(r["held_out"]["iid"]["generated_accuracy"] >= 0.90 for r in oracle),
        random_label_leakage_alarm=any(r["held_out"]["iid"]["generated_accuracy"] > 0.20 for r in noise),
        architecture_qualified=False)
    manifest["phase"] = "complete"
    save()


if __name__ == "__main__":
    main()
