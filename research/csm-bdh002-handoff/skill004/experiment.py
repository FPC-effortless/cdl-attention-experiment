import csv
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "skill002"))
from run import admit, execute, learn_trace, plan

swap = learn_trace(((8, 3, None), (8, 3, 8), (3, 3, 8), (3, 8, 8), (3, 8, None)))
assert admit(swap)
arms = {"primitive": (None, False), "deferred": (swap, True), "eager": (swap, False)}
features = ["misplaced", "reciprocal_pairs", "duplicate_values"]


def roster(n):
    initial = (*range(n), None)
    for target in product(range(n), repeat=n):
        goal = (*target, None)
        misplaced = sum(i != value for i, value in enumerate(target))
        reciprocal = sum(target[i] == j and target[j] == i for i, j in combinations(range(n), 2))
        yield initial, goal, [misplaced, reciprocal, n - len(set(target))]


def costs(initial, goal):
    measured = []
    for args in arms.values():
        actions = plan(initial, goal, *args)
        expanded = tuple(op for action in actions for op in action)
        assert execute(initial, expanded) == goal
        measured.append(4 * len(expanded) + len(actions))
    return measured


def main():
    x, y = [], []
    for n in (3, 4):
        for initial, goal, row in roster(n):
            measured = costs(initial, goal)
            x.append(row)
            y.append(np.subtract(measured, measured[0]))
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x, y)
    print(export_text(tree, feature_names=features), file=sys.stderr)
    print("Best fixed training arm:", list(arms)[np.argmin(np.sum(y, axis=0))], file=sys.stderr)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("goal", "selected", *arms, "learned", "hindsight"))
    for initial, goal, row in roster(5):
        chosen = int(np.argmin(tree.predict([row])[0]))
        measured = costs(initial, goal)
        writer.writerow(("".join(map(str, goal[:-1])), list(arms)[chosen], *measured,
                         measured[chosen], min(measured)))


if __name__ == "__main__":
    main()
