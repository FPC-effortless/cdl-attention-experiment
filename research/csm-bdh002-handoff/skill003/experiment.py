import csv
import platform
import sys
from itertools import permutations, product
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "skill002"))
from run import admit, execute, learn_trace, plan

initial = (0, 1, 2, 3, None)
goals = [(*values, None) for values in product(initial[:-1], repeat=4)]
swap = learn_trace(((8, 3, None), (8, 3, 8), (3, 3, 8), (3, 8, 8), (3, 8, None)))
assert admit(swap)
arms = {"primitive": (None, False), "with_swap": (swap, False), "deferred": (swap, True)}
with (ROOT.parent / "skill002/scores.csv").open() as source:
    prior = {(r["goal"], r["arm"]): (int(r["dispatches"]), int(r["primitives"]))
             for r in csv.DictReader(source)}

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(("goal", "arm", "dispatches", "primitives"))
for goal in goals:
    key = "".join(map(str, goal[:-1]))
    for arm, args in arms.items():
        actions = plan(initial, goal, *args)
        ops = tuple(op for action in actions for op in action)
        assert execute(initial, ops) == goal
        counts = (len(actions), len(ops))
        if arm != "deferred":
            assert counts == prior[key, arm]
        writer.writerow((key, arm, *counts))

if "--time" in sys.argv:
    timer = csv.writer(sys.stderr, lineterminator="\n")
    timer.writerow(("block", "position", "arm", "total_ns", "python", "machine"))
    for block, order in enumerate(list(permutations(arms)) * 5):
        for position, arm in enumerate(order):
            args = arms[arm]
            start = perf_counter_ns()
            for goal in goals:
                plan(initial, goal, *args)
            elapsed = perf_counter_ns() - start
            timer.writerow((block, position, arm, elapsed, platform.python_version(), platform.machine()))
