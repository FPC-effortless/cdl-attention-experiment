"""LRR-003 reproducible seeded raw-text retrieval benchmark.

Run separately from exploration. This file does not modify optimizer
hyperparameters after seeing any confirmatory results.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .attention_raw import train_attention
from .raw_causal import ALIASES, CausalRoleIndexer, generate, train_model, tokenize
from .symbolic_raw import SymbolicRawIndex


def evaluate_arm(model, examples, *, relation_only=False, literal_relation=False) -> dict:
    ok = 0
    abstentions = 0
    admissions = []
    elapsed = []
    for example in examples:
        if isinstance(model, CausalRoleIndexer):
            result = model.route(example.text, relation_only=relation_only,
                                 literal_relation=literal_relation)
        elif isinstance(model, SymbolicRawIndex):
            result = model.route(example.text, relation_only=relation_only,
                                 literal_relation=literal_relation)
        else:
            result = model.route(example.text)
        predicted = result["token_index"]
        ok += int(predicted == example.gold_index)
        abstentions += int(predicted is None)
        if "admissions" in result:
            admissions.append(result["admissions"])
        if "elapsed_total_cpu_us" in result:
            elapsed.append(result["elapsed_total_cpu_us"])
        if predicted is not None and predicted >= len(tokenize(example.text)) - 1:
            raise AssertionError("future token returned")
    return {
        "correct": ok,
        "trials": len(examples),
        "recall_at_1": ok / len(examples),
        "abstentions": abstentions,
        "abstention_rate": abstentions / len(examples),
        "mean_admissions": sum(admissions)/len(admissions) if admissions else None,
        "median_total_cpu_us": statistics.median(elapsed) if elapsed else None,
        "p95_total_cpu_us": sorted(elapsed)[max(0, int(0.95 * len(elapsed))-1)] if elapsed else None,
    }


def run(*, seeds=(21,22,23), lengths=(6,32,128), trials=32,
        steps=700, batch_size=48, output_dir: str | None = None) -> dict:
    torch.set_num_threads(1)
    if not seeds or not lengths or trials < 1:
        raise ValueError("empty scientific matrix")
    report: dict = {
        "experiment": "LRR-003",
        "status": "confirmatory raw templated text retrieval; NOT language-model replacement",
        "device": "CPU, PyTorch single-thread",
        "split": "train excludes ordered alias pairs (0,1) and (2,3); dev=(0,1); TEST=(2,3)",
        "seeds": list(seeds),
        "lengths": list(lengths),
        "trials_per_length_seed": trials,
        "updates": steps,
        "training_documents_per_arm_seed": steps * batch_size,
        "rows": [],
        "training": [],
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        start = time.perf_counter()
        router, router_meta = train_model(seed, steps=steps, batch_size=batch_size)
        router_train_sec = time.perf_counter() - start

        start = time.perf_counter()
        baseline, baseline_meta = train_attention(seed, steps=steps, batch_size=batch_size)
        baseline_train_sec = time.perf_counter() - start

        start = time.perf_counter()
        symbolic = SymbolicRawIndex()
        symbolic.train(seed, steps=steps, batch_size=batch_size)
        symbolic_train_sec = time.perf_counter() - start

        relative_param_gap = abs(router_meta["trainable_parameters"] -
                                 baseline_meta["trainable_parameters"]) / router_meta["trainable_parameters"]
        if relative_param_gap > 0.05:
            raise AssertionError("neural models are not within the 5% preregistered budget")
        report["training"].append({
            "seed": seed,
            "router": router_meta,
            "attention": baseline_meta,
            "symbolic": {
                "trainable_parameters": 0,
                "training_documents": symbolic.training_documents,
                "inferred_equivalence_groups": symbolic.families.unique_roots(),
            },
            "parameter_gap_fraction": relative_param_gap,
            "training_wall_seconds": {
                "router": router_train_sec,
                "attention": baseline_train_sec,
                "symbolic": symbolic_train_sec,
            },
        })
        if output_dir is not None:
            torch.save(router.state_dict(), os.path.join(output_dir, f"lrr003_seed{seed}.pt"))
            torch.save(baseline.state_dict(), os.path.join(output_dir, f"attention003_seed{seed}.pt"))
        for n in lengths:
            test_cases = [generate(40_000_000 + seed * 1_000_000 + n * 1000 + i,
                                   n_facts=n, split="test") for i in range(trials)]
            if any((x.fact_alias, x.question_alias) not in
                   {(aliases[2], aliases[3]) for aliases in ALIASES.values()}
                   for x in test_cases):
                raise AssertionError("TEST alias split was violated")
            arms = {
                "learned": evaluate_arm(router, test_cases),
                "no_entity_key": evaluate_arm(router, test_cases, relation_only=True),
                "literal_relation": evaluate_arm(router, test_cases, literal_relation=True),
                "symbolic_union_find": evaluate_arm(symbolic, test_cases),
                "attention_pointer": evaluate_arm(baseline, test_cases),
            }
            report["rows"].append({"seed": seed, "facts": n, "arms": arms})
        # Challenges have no training target in the current corpus; keep separate.
        challenges = [
            generate(89_000_000 + seed*1_000+i, 16, split="test", update=True)
            for i in range(trials)
        ]
        renamed = [
            generate(91_000_000 + seed*1_000+i, 16, split="test", new_entity=True)
            for i in range(trials)
        ]
        unknown = [
            x.text[:-(len(x.question_alias) + 2)] + " unknownverb ?"
            for x in renamed
        ]
        unknown_results = {
            "learned_false_positive_rate": sum(router.route(t)["token_index"] is not None
                                               for t in unknown)/len(unknown),
            "symbolic_false_positive_rate": sum(symbolic.route(t)["token_index"] is not None
                                                for t in unknown)/len(unknown),
            "attention_forced_answer_rate": sum(baseline.route(t)["token_index"] is not None
                                                for t in unknown)/len(unknown),
        }
        report.setdefault("challenge", []).append({
            "seed": seed,
            "unknown_relation": unknown_results,
            "late_conflicting_update": {
                "learned": evaluate_arm(router, challenges),
                "symbolic": evaluate_arm(symbolic, challenges),
                "attention": evaluate_arm(baseline, challenges),
            },
            "unseen_entity_surface": {
                "learned": evaluate_arm(router, renamed),
                "symbolic": evaluate_arm(symbolic, renamed),
                "attention": evaluate_arm(baseline, renamed),
            },
        })
    primary = [row for row in report["rows"] if row["facts"] == 32]
    if len(primary) != len(seeds):
        raise AssertionError("missing primary rows")
    learned = [row["arms"]["learned"]["recall_at_1"] for row in primary]
    report["gates"] = {
        "preregistered_feasibility_threshold": 0.70,
        "required_passing_seeds": 2,
        "passing_seeds": sum(x >= 0.70 for x in learned),
        "feasibility_pass": sum(x >= 0.70 for x in learned) >= 2,
        "primary_mean_recall": sum(learned)/len(learned),
        "symbolic_primary_mean_recall": sum(
            row["arms"]["symbolic_union_find"]["recall_at_1"] for row in primary)/len(primary),
        "attention_primary_mean_recall": sum(
            row["arms"]["attention_pointer"]["recall_at_1"] for row in primary)/len(primary),
        "does_not_prove_neural_superiority": True,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="21,22,23")
    parser.add_argument("--lengths", default="6,32,128")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lrr003-results.json"))
    parser.add_argument("--checkpoint-dir", default="/tmp/lrr003-checkpoints")
    args = parser.parse_args()
    result = run(
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        lengths=tuple(int(x) for x in args.lengths.split(",")),
        trials=args.trials, steps=args.steps,
        batch_size=args.batch_size, output_dir=args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    primary = [
        {"seed": row["seed"],
         "learned": row["arms"]["learned"]["recall_at_1"],
         "symbolic": row["arms"]["symbolic_union_find"]["recall_at_1"],
         "attention": row["arms"]["attention_pointer"]["recall_at_1"]}
        for row in result["rows"] if row["facts"] == 32]
    print(json.dumps({"gates": result["gates"], "primary": primary}, indent=2))
    print(f"Complete unfiltered result: {args.output}")


if __name__ == "__main__":
    main()
