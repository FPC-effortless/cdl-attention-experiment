"""LRR-004 confirmation: fixed five seeds, new held-out pair and hard negatives.

All outcomes are written even when the preregistered quality gates fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from .raw_causal import ALIASES, IDS, tokenize, examples_to_batch
from .symbolic_raw import SymbolicRawIndex
from .lrr004_locality import (
    make_example, train_arm, KnownLexiconGuard,
    known_absent_challenge, unknown_predicate_challenge,
    distant_predicate_challenge,
)


def score_positive(model, examples, *, relation_only=False, literal_relation=False):
    count = abstained = 0
    offsets = {"question_rel0": 0, "gold_rel0": 0,
               "question_ent1": 0, "gold_ent1": 0, "gold_gate": 0}
    for example in examples:
        result = model.route(
            example.text, relation_only=relation_only,
            literal_relation=literal_relation,
        )
        count += int(result["token_index"] == example.gold_index)
        abstained += int(result["token_index"] is None)
        if result["token_index"] is not None and result["token_index"] >= len(example.tokens) - 1:
            raise AssertionError("future token")
        x, y, q = examples_to_batch([example])
        with torch.inference_mode():
            ent, rel, _, gate, _ = model(x)
        offsets["question_rel0"] += int(rel[0, q.item()].argmax(-1).item() == 0)
        offsets["gold_rel0"] += int(rel[0, y.item()].argmax(-1).item() == 0)
        offsets["question_ent1"] += int(ent[0, q.item()].argmax(-1).item() == 1)
        offsets["gold_ent1"] += int(ent[0, y.item()].argmax(-1).item() == 1)
        offsets["gold_gate"] += int(gate[0, y.item()].sigmoid().item() >= 0.5)
    n = len(examples)
    return {
        "correct": count,
        "total": n,
        "recall_at_1": count/n,
        "abstentions": abstained,
        "abstention_rate": abstained/n,
        "hard_offset_fraction": {name: value/n for name, value in offsets.items()},
    }


def model_diagnostics(model):
    code_ids = model.lexical_code.weight.detach().argmax(-1).tolist()
    codes = {
        family: {word: code_ids[IDS[word]] for word in words}
        for family, words in ALIASES.items()
    }
    return {
        "hard_alias_codes": codes,
        "hard_codes_used_across_aliases": len(set(
            code for members in codes.values() for code in members.values()
        )),
        "families_with_uniform_alias_code": sum(
            len(set(members.values())) == 1 for members in codes.values()
        ),
    }


def train_symbolic_on_matched_stream(seed, *, steps, batch_size):
    symbolic = SymbolicRawIndex()
    for step in range(steps):
        for i in range(batch_size):
            example = make_example(
                seed * 10_000_000 + step * 1_000 + i,
                random.Random(seed + step * 1_000 + i).randint(5, 8),
                split="train",
            )
            symbolic.observe_training(example.text, example.answer)
    return symbolic


def false_positive_count(router, texts):
    n = len(texts)
    affirmative = sum(router.route(text)["token_index"] is not None for text in texts)
    return {"answered": affirmative, "total": n, "false_answer_rate": affirmative/n}


def evaluate_one(seed, n_facts, n_trials, model, lexical_support, symbolic):
    cases = [
        make_example(
            75_000_000 + seed * 1_000_000 + n_facts * 1000 + i,
            n_facts, split="test",
        )
        for i in range(n_trials)
    ]
    for case in cases:
        aliases = ALIASES[case.family]
        if (case.fact_alias, case.question_alias) != (aliases[1], aliases[2]):
            raise AssertionError("confirmatory TEST split invalid")
    guard = KnownLexiconGuard(model, lexical_support)
    positive = score_positive(model, cases)
    guard_correct = sum(
        guard.route(case.text)["token_index"] == case.gold_index for case in cases
    )
    literal = sum(
        model.route(case.text, literal_relation=True)["token_index"] == case.gold_index
        for case in cases
    )
    relation_only = sum(
        model.route(case.text, relation_only=True)["token_index"] == case.gold_index
        for case in cases
    )
    oracle_symbolic = sum(
        symbolic.route(case.text)["token_index"] == case.gold_index
        for case in cases
    )
    unknown = [unknown_predicate_challenge(case) for case in cases]
    absent = [known_absent_challenge(case) for case in cases]
    far = [distant_predicate_challenge(case) for case in cases]
    return {
        "seed": seed, "facts": n_facts,
        "learned": positive,
        "guarded_known_recall": guard_correct/n_trials,
        "symbolic_recall": oracle_symbolic/n_trials,
        "literal_relation_recall": literal/n_trials,
        "no_entity_recall": relation_only/n_trials,
        "unknown_predicate": {
            "raw": false_positive_count(model, unknown),
            "lexical_guard": false_positive_count(guard, unknown),
        },
        "known_relation_absent": {
            "raw": false_positive_count(model, absent),
            "lexical_guard": false_positive_count(guard, absent),
        },
        "nonlocal_adverb_challenge": {
            "correct": sum(
                model.route(text)["token_index"] == gold
                for text, gold in far
            ),
            "total": n_trials,
        },
        "lexical_support_count": len(lexical_support),
        "code_diagnostics": model_diagnostics(model),
    }


def run(seeds=(31, 32, 33, 34, 35), lengths=(6, 32, 128),
        trials=64, steps=700, batch_size=48, outdir=None):
    torch.set_num_threads(1)
    report = {
        "experiment": "LRR-004",
        "interpretation": "raw-template locality probe; no attention replacement",
        "device": "CPU one PyTorch thread",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seeds": list(seeds), "lengths": list(lengths),
        "trials_per_cell": trials,
        "updates_per_model": steps,
        "batch_size": batch_size,
        "arms": {"control": 0.0, "locality": 0.15},
        "training": [], "results": [],
    }
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        symbolic = train_symbolic_on_matched_stream(
            seed, steps=steps, batch_size=batch_size,
        )
        for arm, lam in report["arms"].items():
            model, metadata, supported = train_arm(
                seed, distance_weight=lam, steps=steps, batch_size=batch_size,
            )
            report["training"].append({"arm": arm, **metadata})
            if outdir is not None:
                torch.save(model.state_dict(), Path(outdir)/f"lrr004_{arm}_seed{seed}.pt")
            if len(supported) != sum(len(words) for words in ALIASES.values()):
                raise AssertionError("some query lexemes were not observed in training")
            for n in lengths:
                report["results"].append({
                    "arm": arm,
                    **evaluate_one(seed, n, trials, model, supported, symbolic),
                })
    def row(arm,seed,n):
        hits=[r for r in report["results"] if
              r["arm"]==arm and r["seed"]==seed and r["facts"]==n]
        if len(hits)!=1:
            raise AssertionError("missing or duplicated report")
        return hits[0]
    control = [row("control",seed,32)["learned"]["recall_at_1"] for seed in seeds]
    local = [row("locality",seed,32)["learned"]["recall_at_1"] for seed in seeds]
    delta = [new-old for new,old in zip(local,control)]
    primary = [row("locality",seed,32) for seed in seeds]
    fpr_unk = sum(r["unknown_predicate"]["lexical_guard"]["answered"]
                  for r in primary)/(trials*len(seeds))
    fpr_abs = sum(r["known_relation_absent"]["lexical_guard"]["answered"]
                  for r in primary)/(trials*len(seeds))
    guard_acc = sum(r["guarded_known_recall"] for r in primary)/len(seeds)
    learned_mean = statistics.mean(local)
    report["decision"] = {
        "control_primary_by_seed": control,
        "locality_primary_by_seed": local,
        "paired_deltas": delta,
        "control_mean": statistics.mean(control),
        "locality_mean": learned_mean,
        "pass_optimizer_stability": sum(x>=0.70 for x in local)>=4 and learned_mean>=0.85,
        "pass_causal_improvement": statistics.mean(delta)>=0.10 and all(x>=-0.05 for x in delta),
        "guarded_positive_mean": guard_acc,
        "unknown_predicate_guard_false_answer_rate": fpr_unk,
        "known_absent_guard_false_answer_rate": fpr_abs,
        "pass_lexical_guard": fpr_unk <= 0.05 and learned_mean-guard_acc <= 0.01,
        "pass_known_absent": fpr_abs<=0.10,
        "neural_advantage_over_symbolic_proven": False,
        "language_model_attention_replacement_proven": False,
    }
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",default="31,32,33,34,35")
    ap.add_argument("--lengths",default="6,32,128")
    ap.add_argument("--trials",type=int,default=64)
    ap.add_argument("--steps",type=int,default=700)
    ap.add_argument("--batch-size",type=int,default=48)
    ap.add_argument("--output",type=Path,default=Path("/tmp/lrr004.json"))
    ap.add_argument("--checkpoint-dir",type=Path,default=Path("/tmp/lrr004-checkpoints"))
    a=ap.parse_args()
    result=run(
        seeds=tuple(int(s) for s in a.seeds.split(",")),
        lengths=tuple(int(s) for s in a.lengths.split(",")),
        trials=a.trials,steps=a.steps,batch_size=a.batch_size,
        outdir=str(a.checkpoint_dir),
    )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "decision":result["decision"],
        "training":result["training"],
        "evaluated_cells":len(result["results"]),
        "all_rows_output":str(a.output),
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
