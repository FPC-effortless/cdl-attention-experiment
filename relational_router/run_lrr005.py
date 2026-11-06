"""LRR-005 bounded confirmation with symbolic ceiling and negative controls.

Outputs all failed cells, never drops weak seeds or silently changes thresholds.
"""
from __future__ import annotations
import argparse
import json
import platform
import random
import statistics
from pathlib import Path

import torch

from .lrr005_register import (
    CausalRoleRegister, Example, FAMILIES, NAMES, VOCAB, RESERVED,
    assert_pair_split, fit, make_example, make_training_pair,
)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, word):
        if word not in self.parent:
            self.parent[word] = word
        while self.parent[word] != word:
            self.parent[word] = self.parent[self.parent[word]]
            word = self.parent[word]
        return word

    def unite(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.parent[b] = a

    def count(self):
        return len({self.find(w) for w in self.parent})


KNOWN_PREDICATES = frozenset(word for family in FAMILIES.values() for word in family)


def parse_raw(words):
    """Grammar-aware, privileged symbolic comparator; not learned parsing."""
    chunks = []
    start = 0
    question_start = None
    for i, token in enumerate(words):
        if token == ".":
            span = words[start:i]
            if not span:
                raise ValueError("empty clause")
            preds = [word for word in span if word in KNOWN_PREDICATES]
            if len(preds) != 1:
                raise ValueError("not one recognizable predicate")
            chunks.append((span[0], preds[0], span[-1], i-1))
            start = i+1
    query = words[start:]
    if len(query) < 5 or query[0:2] != ("what", "does") or query[-1] != "?":
        raise ValueError("unsupported question schema")
    return chunks, query[2], query[-2]


class SymbolicRoleIndex:
    def __init__(self):
        self.uf = UnionFind()
        self.training_questions = 0

    def add_example(self, example: Example):
        clauses, _, query_pred = parse_raw(example.words)
        if example.answer < 0:
            raise ValueError("symbolic teacher requires a supported answer")
        relevant = [c for c in clauses if c[-1] == example.answer]
        if len(relevant) != 1:
            raise ValueError("answer position does not identify unique clause")
        self.uf.unite(relevant[0][1], query_pred)
        self.training_questions += 1

    def train(self, seed, steps=500, batch_size=32):
        for step in range(steps):
            for i in range(batch_size//2):
                a,b = make_training_pair(seed*1_000_000+step*1_000+i, step, i)
                self.add_example(a)
                self.add_example(b)

    def route(self, words):
        clauses, entity, predicate = parse_raw(words)
        index = {}
        for name, relation, _, pos in clauses:
            index[(name,self.uf.find(relation))] = pos
        return index.get((entity,self.uf.find(predicate)))


def known_absent(example: Example):
    """Keep known entity and predicate but remove its only matching fact."""
    words = list(example.words)
    answer = example.answer
    start = answer
    while start > 0 and words[start-1] != ".":
        start -= 1
    if words[start] != example.subject:
        raise AssertionError("target clause subject mismatch")
    # Different existing entity, never the query entity.
    existing = sorted({f[0] for f in example.facts if f[0] != example.subject})
    if not existing:
        raise AssertionError("no alternative existing entity")
    words[start] = existing[0]
    clauses, subject, predicate = parse_raw(tuple(words))
    if subject != example.subject:
        raise AssertionError("question entity modified")
    if any(n == subject and pred in FAMILIES[example.family] for n,pred,_,_ in clauses):
        raise AssertionError("negative example still contains a valid fact")
    return tuple(words)


def unknown_relation(example: Example):
    words = list(example.words)
    words[-2] = "unseenverb"
    return tuple(words)


def evaluate_one(model, symbolic, *, seed, n, trials=64):
    outcomes = {}
    for layout in ("regular", "shift", "novel_fillers"):
        cases = [
            make_example(seed * 100_000_000 + n * 1_000 + i,
                         n, split="test", layout=layout)
            for i in range(trials)
        ]
        if not all(case.answer >= 0 for case in cases):
            raise AssertionError("unexpected test gold label")
        for case in cases:
            assert_pair_split(case, training=False)
        hits = 0
        abstained = 0
        symbolic_hits = 0
        writes = []
        for case in cases:
            prediction, diagnostic = model.route(case.words)
            hits += prediction == case.answer
            abstained += prediction is None
            symbolic_hits += symbolic.route(case.words) == case.answer
            writes.append(diagnostic["writes"])
        outcomes[layout] = {
            "correct": hits,
            "trials": trials,
            "recall_at_1": hits / trials,
            "abstentions": abstained,
            "symbolic_correct": symbolic_hits,
            "symbolic_recall": symbolic_hits/trials,
            "mean_index_writes": statistics.mean(writes),
        }
        if layout == "regular":
            false_absent = sum(model.route(known_absent(e))[0] is not None for e in cases)
            false_unknown = sum(model.route(unknown_relation(e))[0] is not None for e in cases)
            symbolic_false_absent = sum(symbolic.route(known_absent(e)) is not None for e in cases)
            symbolic_false_unknown = sum(symbolic.route(unknown_relation(e)) is not None for e in cases)
            outcomes["known_absent"] = {
                "false_answers": false_absent, "trials": trials,
                "false_answer_rate": false_absent/trials,
                "symbolic_false_answers": symbolic_false_absent,
            }
            outcomes["unknown_relation"] = {
                "false_answers": false_unknown, "trials": trials,
                "false_answer_rate": false_unknown/trials,
                "symbolic_false_answers": symbolic_false_unknown,
            }
    return outcomes


def run(*, seeds=(41,42,43,44,45), lengths=(7,32,128),
        trials=64, steps=500, batch=32, checkpoint_dir=None):
    torch.set_num_threads(1)
    output = {
        "experiment": "LRR-005",
        "status": "controlled raw token relation addressing, not a trained language model",
        "device": "CPU",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "seeds": list(seeds),
        "lengths": list(lengths),
        "trials_per_cell": trials,
        "steps": steps,
        "questions_per_step": batch,
        "reserved_test_pair": "(1,2)",
        "development_pair": "(3,0)",
        "codes_supplied": 8,
        "arms": {"register":0.0, "register_local_credit":0.2},
        "training": [],
        "results": [],
    }
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        symbol = SymbolicRoleIndex()
        symbol.train(seed,steps,batch)
        for arm, credit in output["arms"].items():
            model, meta = fit(seed,steps=steps,batch_size=batch,credit=credit)
            output["training"].append({"arm":arm,**meta,
                                       "symbolic_equivalence_groups":symbol.uf.count()})
            if checkpoint_dir:
                torch.save(model.state_dict(),Path(checkpoint_dir)/f"lrr005_{arm}_{seed}.pt")
            for length in lengths:
                outcome = evaluate_one(
                    model, symbol, seed=seed + 50_000, n=length, trials=trials)
                output["results"].append({"arm":arm,"seed":seed,"facts":length,"outcomes":outcome})
    primary = [x for x in output["results"] if x["facts"] == 32]
    def select(arm):
        return [next(x for x in primary if x["seed"]==seed and x["arm"]==arm)
                for seed in seeds]
    a = select("register")
    b = select("register_local_credit")
    a_scores = [r["outcomes"]["regular"]["recall_at_1"] for r in a]
    b_scores = [r["outcomes"]["regular"]["recall_at_1"] for r in b]
    delta = [y-x for x,y in zip(a_scores,b_scores)]
    shift_mean = statistics.mean(r["outcomes"]["shift"]["recall_at_1"] for r in b)
    negative_known = statistics.mean(r["outcomes"]["known_absent"]["false_answer_rate"] for r in b)
    negative_unknown = statistics.mean(r["outcomes"]["unknown_relation"]["false_answer_rate"] for r in b)
    output["decision"] = {
        "register_primary_by_seed":a_scores,
        "credit_primary_by_seed":b_scores,
        "paired_deltas":delta,
        "credit_primary_mean":statistics.mean(b_scores),
        "register_primary_mean":statistics.mean(a_scores),
        "credit_capability_gate":sum(v>=.80 for v in b_scores)>=4 and statistics.mean(b_scores)>=.85 and
                                 statistics.mean(delta)>=.10 and min(delta)>=-.05,
        "shift_recall_mean":shift_mean,
        "shift_gate":shift_mean>=.70,
        "known_absent_false_answer_mean":negative_known,
        "known_absent_gate":negative_known<=.10,
        "unknown_relation_false_answer_mean":negative_unknown,
        "unknown_relation_gate":negative_unknown<=.10,
        "neural_superiority_proven":False,
        "language_model_attention_replacement_proven":False,
    }
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--trials",type=int,default=64)
    parser.add_argument("--seeds",default="41,42,43,44,45")
    parser.add_argument("--lengths",default="7,32,128")
    parser.add_argument("--output",default="/tmp/lrr005-results.json")
    parser.add_argument("--checkpoints",default="/tmp/lrr005-weights")
    args=parser.parse_args()
    results=run(seeds=tuple(map(int,args.seeds.split(","))),
                lengths=tuple(map(int,args.lengths.split(","))),
                steps=args.steps,trials=args.trials,checkpoint_dir=args.checkpoints)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"decision":results["decision"],"training":results["training"],
                      "total_result_cells":len(results["results"])},indent=2))
    print("Full raw report:",args.output)


if __name__=="__main__":
    main()
