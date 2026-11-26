"""Run only the prospectively specified optional numeric component diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from fusion.composition import RecordComputer


ROOT = Path(__file__).resolve().parent
SEEDS = (1701, 1702, 1703)
STEPS, BATCH, D = 1000, 64, 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def generate(generator, count):
    return torch.rand(count, 2, D, generator=generator) * 1.5 - 0.75


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/composition")
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = {"protocol": "COMPOSITION_PROTOCOL.md", "seeds": list(SEEDS), "steps": STEPS,
            "batch": BATCH, "value_dim": D, "hidden_dim": 32,
            "torch": str(torch.__version__), "threads": 1,
            "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
                ROOT / "COMPOSITION_PROTOCOL.md", ROOT / "train_composition.py", ROOT / "fusion/composition.py")}}
    lockpath = args.output / "PROSPECTIVE_LOCK.json"
    if args.lock_only:
        if lockpath.exists():
            raise RuntimeError("prospective lock already exists")
        write(lockpath, lock)
        print(json.dumps({"status": "protocol_locked", "path": str(lockpath), "sha256": sha(lockpath)}))
        return
    if not lockpath.exists() or json.loads(lockpath.read_text()) != lock:
        raise RuntimeError("write and review the prospective lock before fitting; sources must still match")
    if (args.output / "RESULTS.json").exists():
        raise RuntimeError("results already exist; do not overwrite or silently rerun a held-out test")
    test = generate(torch.Generator().manual_seed(99170401), 4096)
    target = test.mean(1)
    test_hash = hashlib.sha256(test.numpy().tobytes()).hexdigest()
    results = {"protocol_sha256": sha(ROOT / "COMPOSITION_PROTOCOL.md"),
               "lock_sha256": sha(lockpath), "test_input_sha256": test_hash,
               "bayes_reset_expected_mse": 0.046875, "numeric_only": True, "seeds": []}
    for seed in SEEDS:
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        torch.manual_seed(seed)
        connected = RecordComputer(D, 32)
        reset = copy.deepcopy(connected)
        models = {"connected": connected, "reset_control": reset}
        initial = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in connected.parameters())).hexdigest()
        optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
                      for name, model in models.items()}
        generator = torch.Generator().manual_seed(900000 + seed)
        train_hash = hashlib.sha256()
        times = {name: {"wall_seconds": 0.0, "cpu_seconds": 0.0} for name in models}
        trace = []
        completed = 0
        for step in range(1, STEPS + 1):
            if time.process_time() - started_cpu > 300:
                break
            inputs = generate(generator, BATCH)
            train_hash.update(inputs.numpy().tobytes())
            truth = inputs.mean(1)
            losses = {}
            for name, model in models.items():
                cpu, wall = time.process_time(), time.perf_counter()
                optimizers[name].zero_grad(set_to_none=True)
                predicted = model(inputs[:, 0], inputs[:, 1], reset_between=name == "reset_control")
                loss = (predicted - truth).square().mean()
                loss.backward(); optimizers[name].step()
                times[name]["cpu_seconds"] += time.process_time() - cpu
                times[name]["wall_seconds"] += time.perf_counter() - wall
                losses[name] = float(loss.detach())
            completed = step
            if step == 1 or step % 100 == 0:
                trace.append({"step": step, "loss": losses})
        outputs = {}
        with torch.no_grad():
            for name, model in models.items():
                model.eval()
                outputs[name] = model(test[:, 0], test[:, 1], reset_between=name == "reset_control")
            outputs["acute_reset"] = connected(test[:, 0], test[:, 1], reset_between=True)
            outputs["permuted_first"] = connected(test[:, 0].roll(1, 0), test[:, 1])
            outputs["second_only_optimal"] = test[:, 1] / 2
        metrics = {name: float((pred - target).square().mean()) for name, pred in outputs.items()}
        first = test[:128, 0].clone().requires_grad_(True)
        value = connected(first, test[:128, 1])
        gradient, = torch.autograd.grad(value.sum(), first)
        gradient_summary = {"finite": bool(torch.isfinite(gradient).all()),
                            "abs_mean": float(gradient.abs().mean()),
                            "mean": float(gradient.mean()), "abs_max": float(gradient.abs().max())}
        seedpath = args.output / f"seed{seed}"
        seedpath.mkdir()
        checkpoint_hashes = {}
        for name, model in models.items():
            path = seedpath / f"{name}.pt"
            torch.save(model.checkpoint_payload(seed=seed, arm=name, steps=completed,
                       batch=BATCH, protocol_sha256=results["protocol_sha256"],
                       training_inputs_sha256=train_hash.hexdigest()), path)
            checkpoint_hashes[name] = sha(path)
        torch.save({"inputs": test, "target": target, "outputs": outputs}, seedpath / "heldout_outputs.pt")
        gates = {"complete": completed == STEPS, "connected_mse": metrics["connected"] < .002,
                 "tenfold_control_gain": metrics["reset_control"] / metrics["connected"] >= 10,
                 "acute_reset": metrics["acute_reset"] > .03,
                 "first_input_gradient": gradient_summary["finite"] and gradient_summary["abs_mean"] > 0}
        result = {"seed": seed, "completed_updates_per_arm": completed,
                  "training_pairs_per_arm": completed * BATCH,
                  "parameter_count": sum(p.numel() for p in connected.parameters()),
                  "controller_state_bytes": 32 * 4, "initial_weights_sha256": initial,
                  "training_inputs_sha256": train_hash.hexdigest(), "mse": metrics,
                  "gradient": gradient_summary, "gates": gates, "pass": all(gates.values()),
                  "fit_times": times, "paired_cpu_seconds": time.process_time() - started_cpu,
                  "paired_wall_seconds": time.perf_counter() - started_wall,
                  "checkpoint_sha256": checkpoint_hashes, "trace": trace}
        write(seedpath / "RESULT.json", result)
        results["seeds"].append(result)
        print(json.dumps({"seed": seed, "mse": metrics, "pass": result["pass"],
                          "paired_cpu_seconds": result["paired_cpu_seconds"]}), flush=True)
    results["all_numeric_gates_pass"] = all(result["pass"] for result in results["seeds"])
    results["qualification"] = "bounded numeric computation component only; no joint language/retrieval qualification"
    write(args.output / "RESULTS.json", results)


if __name__ == "__main__":
    main()
