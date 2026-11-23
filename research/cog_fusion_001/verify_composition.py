"""Prospective first-16 durable check of each numeric endpoint, without selection."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

from fusion.composition import RecordComputer, RecordRuntime, StaleDerivedError
from fusion.store import InjectedFailure

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/composition"
SEEDS = (1701, 1702, 1703)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    torch.set_num_threads(1)
    protocol = {"selection": "first 16 common heldout pairs per seed, no outcome selection",
                "seeds": list(SEEDS), "first_indices": list(range(16)),
                "odd_indices": "restart after first read before consuming second",
                "all_indices": "restart after derived publication; clear controller; validate derived parents",
                "neural_output_tolerance": 1e-6, "erasure_cases": {"first_source": 0, "second_source": 15},
                "qualification": "typed supplied-address numerical durability only"}
    lock = OUT / "DURABLE_PROTOCOL.json"
    if not lock.exists() or json.loads(lock.read_text()) != protocol:
        raise RuntimeError("durable protocol must be written before results are consumed")
    results = {"protocol_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "seeds": []}
    for seed in SEEDS:
        seed_dir = OUT / f"seed{seed}"
        fitted = RecordComputer.from_checkpoint(seed_dir / "connected.pt").eval()
        evaluation = torch.load(seed_dir / "heldout_outputs.pt", weights_only=True, map_location="cpu")
        database = seed_dir / "durable.sqlite"
        if database.exists():
            raise RuntimeError("do not overwrite a prior durable evaluation")
        runtime = RecordRuntime(fitted, database, capacity=128)
        protected = ("protected", "untouched", "vector")
        runtime.put(protected, [.125] * 8, "protected-source")
        protected_packet = runtime.store.resolve(protected)
        rows = []
        for index in range(16):
            prefix = f"pair{index}"
            first, second, destination = [(prefix, x, "vector") for x in ("first", "second", "result")]
            inputs = evaluation["inputs"][index]
            runtime.put(first, inputs[0].tolist(), f"{prefix}-source-a")
            runtime.put(second, inputs[1].tolist(), f"{prefix}-source-b")
            intermediate_equal = None
            if index % 2:
                runtime.begin_read(first, f"{prefix}-begin")
                intermediate = runtime.state["record_controller"].clone()
                runtime.close(); runtime = RecordRuntime(fitted, database)
                intermediate_equal = torch.equal(intermediate, runtime.state["record_controller"])
                receipt = runtime.finish_read(second, destination, f"{prefix}-finish")
            else:
                receipt = runtime.compute(first, second, destination, f"{prefix}-compute")
            expected = evaluation["outputs"]["connected"][index]
            actual = torch.tensor(runtime.read_derived(destination))
            maximum_error = float((actual - expected).abs().max())
            controller = runtime.state["record_controller"].clone()
            runtime.close(); runtime = RecordRuntime(fitted, database)
            controller_equal = torch.equal(controller, runtime.state["record_controller"])
            before_reset = runtime.read_derived(destination)
            runtime.reset_controller(f"{prefix}-controller-reset")
            after_reset = runtime.read_derived(destination)
            row = {"index": index, "maximum_absolute_output_difference": maximum_error,
                   "restart_between_reads": bool(index % 2), "intermediate_controller_equal": intermediate_equal,
                   "postcommit_controller_equal": controller_equal,
                   "derived_value_survives_controller_reset": before_reset == after_reset,
                   "untouched_record_preserved": runtime.store.validate_packet(protected_packet),
                   "input": inputs.tolist(), "stored_output": actual.tolist(),
                   "target": evaluation["target"][index].tolist()}
            row["pass"] = (maximum_error <= 1e-6 and controller_equal
                           and intermediate_equal is not False and before_reset == after_reset
                           and row["untouched_record_preserved"])
            rows.append(row)
        erasures = []
        for index, suffix in ((0, "a"), (15, "b")):
            runtime.store.invalidate_source(f"pair{index}-source-{suffix}", request_id=f"erase-{index}-{suffix}")
            blocked = False
            try:
                runtime.read_derived((f"pair{index}", "result", "vector"))
            except StaleDerivedError:
                blocked = True
            erasures.append({"index": index, "source": suffix, "invalidated": blocked})
        assert len(runtime.read_derived(("pair1", "result", "vector"))) == 8
        snapshot = runtime.store.snapshot_state()
        try:
            runtime.compute(("pair1", "first", "vector"), ("pair1", "second", "vector"),
                            ("fault", "result", "vector"), "fault-before", failpoint="before_commit")
        except InjectedFailure:
            pass
        rollback = runtime.store.snapshot_state() == snapshot
        result = {"seed": seed, "rows": rows, "source_erasures": erasures,
                  "before_commit_rollback": rollback,
                  "pass": all(r["pass"] for r in rows) and all(e["invalidated"] for e in erasures) and rollback}
        runtime.close()
        write(seed_dir / "DURABLE_RESULT.json", result)
        results["seeds"].append(result)
        print(json.dumps({"seed": seed, "durable_cases": len(rows), "pass": result["pass"]}), flush=True)
    results["pass"] = all(seed["pass"] for seed in results["seeds"])
    write(OUT / "DURABLE_RESULTS.json", results)


if __name__ == "__main__":
    main()
