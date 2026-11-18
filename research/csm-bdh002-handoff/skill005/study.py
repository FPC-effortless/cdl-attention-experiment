import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "skill004"))
from experiment import arms, costs, roster


def independent_pairs(goal):
    target = goal[:-1]
    return sum(i < j and target[j] == i and target.count(i) == target.count(j) == 1
               for i, j in enumerate(target))


def fit():
    rows, targets = [], []
    for size in (3, 4):
        for initial, goal, row in roster(size):
            measured = costs(initial, goal)
            rows.append(row + [independent_pairs(goal)])
            targets.append([value - measured[0] for value in measured])
    models = {}
    for name, width in (("original", 3), ("independent", 4)):
        tree = DecisionTreeRegressor(max_depth=3, random_state=0)
        tree.fit([row[:width] for row in rows], targets)
        nodes = tree.tree_
        models[name] = [
            dict(left=int(nodes.children_left[i]), right=int(nodes.children_right[i]),
                 feature=int(nodes.feature[i]), threshold=float(nodes.threshold[i]),
                 arm=int(nodes.value[i].argmin()))
            for i in range(nodes.node_count)
        ]
    return models


def choose(nodes, row):
    node = nodes[0]
    while node["left"] >= 0:
        side = "left" if row[node["feature"]] <= node["threshold"] else "right"
        node = nodes[node[side]]
    return node["arm"]


def evaluate():
    models = json.loads((ROOT / "models.json").read_text())
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("goal", *arms, "original", "independent", "selected", "hindsight"))
    for initial, goal, row in roster(6):
        row.append(independent_pairs(goal))
        original = choose(models["original"], row)
        selected = choose(models["independent"], row)
        measured = costs(initial, goal)
        writer.writerow(("".join(map(str, goal[:-1])), *measured, measured[original],
                         measured[selected], list(arms)[selected], min(measured)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "evaluate"))
    if parser.parse_args().mode == "fit":
        print(json.dumps(fit(), indent=2))
    else:
        evaluate()
