"""Compare constructive planners on the complete four-record assignment space."""
import csv
import sys
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill001"))
from pilot import admit, bind, execute, learn_trace


def safe_copy(state, goal, pending, required):
    for i in pending:
        if state[i] not in required or state.count(state[i]) > 1:
            return ((i, state.index(goal[i])),)
    return None


def plan(initial, goal, swap, defer_swap=False):
    state = initial
    scratch = len(goal) - 1
    visited = set()
    actions = []
    while state != goal:
        assert state not in visited
        visited.add(state)
        pending = [i for i in range(scratch) if state[i] != goal[i]]
        required = {goal[i] for i in pending}
        action = None
        if defer_swap or not swap:
            action = safe_copy(state, goal, pending, required)
        if action is None and swap and state[scratch] is None:
            for i, j in combinations(pending, 2):
                if (state[j], state[i]) == (goal[i], goal[j]):
                    action = bind(swap, (i, j, scratch))
                    break
        if action is None and swap and not defer_swap:
            action = safe_copy(state, goal, pending, required)
        if action is None:
            if pending:
                assert state[scratch] not in required
                action = ((scratch, pending[0]),)
            else:
                action = ((scratch, None),)
        state = execute(state, action)
        assert state is not None
        actions.append(action)
    return actions


def main():
    swap = learn_trace(((8, 3, None), (8, 3, 8), (3, 3, 8), (3, 8, 8), (3, 8, None)))
    assert admit(swap)
    initial = (0, 1, 2, 3, None)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("goal", "family", "arm", "dispatches", "primitives"))
    for target in product(initial[:-1], repeat=4):
        goal = (*target, None)
        for arm, skill in (("primitive", None), ("with_swap", swap)):
            actions = plan(initial, goal, skill)
            expanded = tuple(op for action in actions for op in action)
            assert execute(initial, expanded) == goal
            family = "permutation" if len(set(target)) == 4 else "repeated"
            writer.writerow(("".join(map(str, target)), family, arm, len(actions), len(expanded)))


if __name__ == "__main__":
    main()
