"""Summarize BALANCE readouts grouped by preset, scale, budget and seed."""
import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

ARMS = tuple("ABCDEFGH") + ("GQA", "GATED_DELTA", "GRU", "MAMBA2")
SEEDS = (9731, 9743, 9769)
BUDGETS = ("matched_data", "matched_cpu")


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def average(rows):
    return {k: stats.mean(r[k] for r in rows) for k in rows[0]}


def row_metrics(record):
    p = record["profile"]
    numbers = {
        "dev_bpb": record["dev_bpb"],
        "decode_step_ms": p["decode_step_median_ms"],
        "decode_tokens_s": p["decode_tokens_s"],
        "state_backing_bytes": p["final_carried_tensor_state"]["backing_tensor_bytes"],
        "train_cpu_s": record["train_cpu_seconds"],
        "train_wall_s": record["train_wall_seconds"],
        "train_text_bytes_s": record["train_language_bytes_per_wall_second"],
        "updates": record["step"],
    }
    for name, result in sorted(record["panels"].items()):
        if not 0 <= result["both_correct"] <= result["worlds"] or result["worlds"] <= 0:
            raise ValueError(f"invalid paired count for {name}")
        numbers[name + "_paired"] = result["both_correct"] / result["worlds"]
        numbers[name + "_brier"] = result["multiclass_brier"]
    if not all(math.isfinite(v) for v in numbers.values()):
        raise ValueError("nonfinite metric")
    return numbers


def signature(run):
    return encode({
        "budgets": run["budgets"], "source_commit": run["source_commit"],
        "threads": run["threads"], "corpus": run["corpus"], "torch": run["torch"],
    })


def validate(records, budget, setup):
    panels = set()
    dev = set()
    training = defaultdict(set)
    for (arm, seed), record in records.items():
        dev.add((record["dev_sha256"], record["dev_byte_targets"]))
        panels.add(encode({k: (v["sha256"], v["worlds"]) for k, v in record["panels"].items()}))
        training[(seed, record["step"])].add(record["training_sha256"])
        nominal = setup["budgets"]
        if budget == "matched_data" and record["step"] != nominal["steps"]:
            raise ValueError(f"{arm}/{seed}: wrong update count")
        if budget == "matched_cpu" and record["train_cpu_seconds"] < nominal["cpu_seconds"]:
            raise ValueError(f"{arm}/{seed}: below CPU budget")
    if len(dev) != 1 or len(panels) != 1 or any(len(s) != 1 for s in training.values()):
        raise ValueError("panel, dev text or within-seed same-step training digests disagree")
    return {
        "dev_sha256": next(iter(dev))[0], "panel_digests_and_worlds": json.loads(next(iter(panels))),
        "training_sha256_by_seed_step": {
            f"{seed}/{step}": next(iter(digests))
            for (seed, step), digests in sorted(training.items())
        },
    }


def pareto(rows, max_state_bytes, max_step_ms):
    # Compare complete, seed-matched observed arms only. Smaller is better
    # for BPB/latency, larger for strict paired retrieval.
    eligible = {}
    for arm, r in rows.items():
        if max_state_bytes is not None and r["state_backing_bytes"] > max_state_bytes:
            continue
        if max_step_ms is not None and r["decode_step_ms"] > max_step_ms:
            continue
        eligible[arm] = (r["dev_bpb"], r["decode_step_ms"], -r["retrieval_paired"])
    ranks = {}
    for arm, point in eligible.items():
        ranks[arm] = sorted(other for other, rival in eligible.items() if other != arm
                            and all(x <= y for x, y in zip(rival, point))
                            and any(x < y for x, y in zip(rival, point)))
    return {
        "frontier": sorted(k for k, dominators in ranks.items() if not dominators),
        "dominators": ranks, "constraints": {"mean_state_bytes_at_most": max_state_bytes,
                                             "mean_decode_step_ms_at_most": max_step_ms},
        "axes": ["minimize dev BPB", "minimize median-step latency", "maximize paired retrieval"],
    }


def summarize(root, expected_arms, expected_seeds, expected_scales,
              max_state_bytes=None, max_step_ms=None):
    setups = defaultdict(dict)
    for path in sorted(Path(root).rglob("metrics.json")):
        r = json.loads(path.read_text())
        arm, seed, scale = r["arm"], r["seed"], r["scale"]
        if arm not in expected_arms or seed not in expected_seeds or scale not in expected_scales:
            continue
        group = setups[signature(r)]
        for name, record in r["readouts"].items():
            if name not in BUDGETS:
                raise ValueError(f"unknown budget {name}")
            key = (scale, name, arm, seed)
            row = (record, r["parameters"])
            if key in group and group[key] != row:
                raise ValueError(f"conflicting duplicate {key} in {path}")
            group[key] = row
    report = {"arms": expected_arms, "seeds": expected_seeds,
              "expected_scales": expected_scales, "experiments": []}
    for encoded_setup, group in setups.items():
        setup = json.loads(encoded_setup)
        experiment = {"settings": setup, "groups": []}
        for scale in expected_scales:
            for budget in BUDGETS:
                records = {(a, s): r for (sc, b, a, s), (r, _) in group.items()
                           if (sc, b) == (scale, budget)}
                missing = {a: [s for s in expected_seeds if (a, s) not in records] for a in expected_arms}
                missing = {a: s for a, s in missing.items() if s}
                item = {"scale": scale, "budget": budget, "complete": not missing,
                        "pending_by_arm": missing, "num_readouts": len(records)}
                experiment["groups"].append(item)
                if not records:
                    continue
                item["hash_checks"] = validate(records, budget, setup)
                observed = {k: row_metrics(v) for k, v in records.items()}
                summaries = {}
                complete = {}
                for arm in expected_arms:
                    rows = {s: observed[arm, s] for s in expected_seeds if (arm, s) in observed}
                    if rows:
                        mean = average(list(rows.values()))
                        summaries[arm] = {"mean": mean, "by_seed": rows,
                                          "parameters": group[scale, budget, arm, next(iter(rows))][1]}
                        if len(rows) == len(expected_seeds):
                            complete[arm] = mean
                item["arm_metrics"] = summaries
                if budget == "matched_data":
                    item["pareto_complete_arms_only"] = pareto(complete, max_state_bytes, max_step_ms)
                contrasts = {}
                for arm in tuple("ABCDEFGH"):
                    for control in ("A", "B", "GQA", "GATED_DELTA", "GRU", "MAMBA2"):
                        if arm == control:
                            continue
                        diffs = {}
                        for s in expected_seeds:
                            if (arm,s) in observed and (control,s) in observed:
                                left, right = observed[arm,s], observed[control,s]
                                diffs[s] = {k: left[k]-right[k] for k in left}
                        if diffs:
                            contrasts[f"{arm} minus {control}"] = {
                                "seed_differences": diffs, "mean_difference": average(list(diffs.values())),
                            }
                item["paired_arm_differences"] = contrasts
        report["experiments"].append(experiment)
    report["status"] = ("complete" if setups and all(g["complete"] for e in report["experiments"]
                          for g in e["groups"]) else "pending")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--scales", nargs="+", default=("1m", "10m"), choices=("1m", "10m"))
    p.add_argument("--max-state-bytes", type=int)
    p.add_argument("--max-step-ms", type=float)
    args = p.parse_args()
    if not args.input.is_dir():
        p.error("input must be an existing directory")
    report = summarize(args.input, args.arms, args.seeds, args.scales,
                       args.max_state_bytes, args.max_step_ms)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Status:", report["status"])
    for n, experiment in enumerate(report["experiments"], 1):
        for g in experiment["groups"]:
            print(f"Experiment {n} {g['scale']} {g['budget']}: {g['num_readouts']} readouts")
            if "arm_metrics" not in g:
                continue
            print("| Arm | Seeds | BPB | Retrieval paired | Held composition paired | Median step ms |")
            print("|---|---:|---:|---:|---:|---:|")
            for a, row in g["arm_metrics"].items():
                m = row["mean"]
                print(f"| {a} | {len(row['by_seed'])} | {m['dev_bpb']:.4f} | "
                      f"{m['retrieval_paired']:.2%} | {m['composition_held_paired']:.2%} | "
                      f"{m['decode_step_ms']:.3f} |")


if __name__ == "__main__":
    main()
