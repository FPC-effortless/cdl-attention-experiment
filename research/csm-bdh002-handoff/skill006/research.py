"""Train the SKILL-006 cost router and evaluate a frozen instance."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import permutations, product
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
if not (HERE / "run.py").exists():
    sys.path.extend(str(HERE.parent / name) for name in ("skill002", "skill001"))
from diagnostic import infer
from run import bind, execute, plan
from pilot import learn_trace

ARMS = ("primitive", "swap", "rotate")
FEATURES = ("misplaced", "duplicates", "two_cycles", "three_cycles")


def acquire():
    swap = learn_trace(((8,3,None),(8,3,8),(3,3,8),(3,8,8),(3,8,None)))
    rotate = infer(((8,3,5,None),(8,3,5,8),(3,3,5,8),(3,5,5,8),(3,5,8,8),(3,5,8,None)))
    return (((), 0), (swap, 2), (rotate, 3))


def features(target):
    pairs = triples = 0
    for i, j in enumerate(target):
        k = target[j]
        pairs += i < j and k == i
        triples += i < j and i < k and j != k and target[k] == i
    return [sum(i != v for i, v in enumerate(target)), len(target) - len(set(target)), pairs, triples]


def probe(initial, goal, program, arity):
    state = initial
    calls = []
    candidates = evaluations = 0
    if program:
        for slots in permutations(range(len(initial) - 1), arity):
            candidates += 1
            if any(initial[i] == goal[i] for i in slots):
                continue
            evaluations += 1
            action = bind(program, (*slots, len(initial) - 1))
            successor = execute(initial, action)
            if all(successor[i] == goal[i] for i in slots):
                state = successor
                calls.append(action)
                break
    calls.extend(plan(state, goal, None))
    ops = tuple(op for call in calls for op in call)
    if execute(initial, ops) != goal:
        raise ValueError(f"Incorrect transfer on {goal}")
    return 4 * len(ops) + len(calls), candidates, evaluations


def fit(seed):
    skills = acquire()
    rows, targets = [], []
    for size in (3, 4, 5):
        initial = (*range(size), None)
        for target in product(range(size), repeat=size):
            costs = [probe(initial, (*target, None), *skill)[0] for skill in skills]
            rows.append(features(target))
            targets.append([value - costs[0] for value in costs])
    estimator = DecisionTreeRegressor(max_depth=3, random_state=seed, criterion="squared_error")
    estimator.fit(rows, targets)
    tree = estimator.tree_
    nodes = [dict(left=int(tree.children_left[i]), right=int(tree.children_right[i]),
                  feature=int(tree.feature[i]), threshold=float(tree.threshold[i]),
                  costs=tree.value[i].reshape(-1).tolist()) for i in range(tree.node_count)]
    return {"seed":seed, "features":list(FEATURES), "arms":list(ARMS), "nodes":nodes,
            "programs":json.loads(json.dumps(skills)), "tree_depth":tree.max_depth,
            "observations":{"features":len(rows[0]), "tree_depth":tree.max_depth, "tree_nodes":tree.node_count,
                            "fit_calls":1, "train_goals":len(rows), "arms":len(targets[0]),
                            "training_labels":sum(map(len,targets))}}


def choose(nodes, values):
    node = nodes[0]
    comparisons = 0
    while node["left"] != -1:
        comparisons += 1
        side = "left" if values[node["feature"]] <= node["threshold"] else "right"
        node = nodes[node[side]]
    return int(np.argmin(node["costs"])), comparisons


def evaluate(model, size, output):
    initial = (*range(size), None)
    totals, selections = Counter(), Counter()
    families = defaultdict(Counter)
    with (output / "scores.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["goal","family","selected","longest_selected",*ARMS,"learned","longest","hindsight",
                         "tree_comparisons",*[f"{arm}_bindings" for arm in ARMS],
                         *[f"{arm}_candidate_executions" for arm in ARMS]])
        for target in product(range(size), repeat=size):
            row = features(target)
            selected, comparisons = choose(model["nodes"], row)
            longest = 2 if row[3] else 1 if row[2] else 0
            arms = [probe(initial, (*target,None), *skill) for skill in model["programs"]]
            costs, bindings, executions = zip(*arms)
            values = (*costs, costs[selected], costs[longest], min(costs))
            labels = (*ARMS,"learned","longest","hindsight")
            family = "repeated" if row[1] else "permutation"
            writer.writerow(["".join(map(str,target)), family, ARMS[selected], ARMS[longest],
                             *values, comparisons, *bindings, *executions])
            totals.update(dict(zip(labels, values)))
            families[family].update(dict(zip(labels, values)))
            families[family]["goals"] += 1
            selections[ARMS[selected]] += 1
    summary = dict(costs=totals, families=families, selections=selections)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return dict(totals), sum(selections.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    seed = int(os.environ.get("TRAINLAB_SEED","0"))
    model = fit(seed)
    if args.mode == "fit":
        args.model.write_text(json.dumps(model, indent=2)+"\n")
        return
    frozen = json.loads(args.model.read_text())
    if model != frozen:
        raise ValueError("Refit on development data disagrees with frozen model")
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    metrics, count = evaluate(frozen, 6, output)
    (output/"metrics.json").write_text(json.dumps(metrics))
    from admission import observe
    observe(output, "research.py", {**model["observations"], "heldout_goals":count})


if __name__ == "__main__":
    main()
