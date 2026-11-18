"""Bounded program induction over a language-free record machine."""
import json
from collections import deque
from itertools import combinations, permutations, product


def primitives(size):
    scratch = size - 1
    return tuple(permutations(range(size), 2)) + ((scratch, None),)


def execute(state, program):
    result = list(state)
    for destination, source in program:
        if source is not None and result[source] is None:
            return None
        result[destination] = None if source is None else result[source]
    return tuple(result)


def synthesize(initial, goal, actions, budget):
    if initial == goal:
        return (), 0, 0
    queue = deque([(initial, ())])
    seen = {initial}
    candidates = operations = 0
    while queue:
        state, plan = queue.popleft()
        for action in actions:
            if operations + len(action) > budget:
                return None, candidates, operations
            candidates += 1
            operations += len(action)
            successor = execute(state, action)
            if successor == goal:
                return plan + (action,), candidates, operations
            if successor is not None and successor not in seen:
                seen.add(successor)
                queue.append((successor, plan + (action,)))
    return None, candidates, operations


def learn_trace(trace):
    actions = primitives(len(trace[0]))
    program = []
    for before, after in zip(trace, trace[1:]):
        matches = [action for action in actions if execute(before, (action,)) == after]
        (action,) = matches
        program.append(action)
    return tuple(program)


def bind(program, records):
    return tuple(
        (records[destination], None if source is None else records[source])
        for destination, source in program
    )


def admit(program):
    # Exercise aliasing of values, scratch cleanup, and preservation of an unrelated record.
    for left, right, other in product((-7, 0, 14), repeat=3):
        for first, second in permutations(range(3), 2):
            initial = [other, other, other, None]
            initial[first], initial[second] = left, right
            expected = initial.copy()
            expected[first], expected[second] = right, left
            if execute(initial, bind(program, (first, second, 3))) != tuple(expected):
                return False
    return True


def main():
    budgets = (512, 2048, 8192, 32768)
    primitive_actions = tuple((op,) for op in primitives(5))
    small_actions = tuple((op,) for op in primitives(3))
    seed_plan, seed_candidates, seed_ops = synthesize((0, 1, None), (1, 0, None), small_actions, budgets[0])
    assert seed_plan is not None
    # Intermediate states from one demonstration, with anonymous record/value IDs.
    trace = ((8, 3, None), (8, 3, 8), (3, 3, 8), (3, 8, 8), (3, 8, None))
    program = learn_trace(trace)
    assert admit(program)
    assert not admit(program[:-1])
    macros = tuple(bind(program, (*pair, 4)) for pair in combinations(range(4), 2))
    tasks = []
    for order in permutations(range(4)):
        initial = tuple(101 + 17 * index for index in order) + (None,)
        a, b, c, d, _ = initial
        tasks.extend(((initial, (b, c, a, d, None)), (initial, (b, a, d, c, None))))
    report = {
        "primitive_budgets_per_task": budgets,
        "outcome_only_seed_candidate_checks": seed_candidates,
        "outcome_only_seed_primitive_checks": seed_ops,
        "demonstration_transitions": len(trace) - 1,
        "demonstration_candidate_checks": (len(trace) - 1) * len(small_actions),
        "admission_cases": 162,
        "transfer_tasks_per_arm": len(tasks),
        "transfer": {},
    }
    # The flat arm retains action-type preferences from the same demonstration.
    def action_type(action, scratch):
        destination, source = action
        return destination == scratch, source == scratch, source is None

    trace_priority = list(dict.fromkeys(action_type(op, 2) for op in program))
    flat_actions = tuple(sorted(
        primitive_actions, key=lambda action: trace_priority.index(action_type(action[0], 4))
    ))
    arms = (
        ("outcome_only", primitive_actions),
        ("demonstrations_flat", flat_actions),
        ("demonstrations_procedural_memory", macros + flat_actions),
    )
    for budget in budgets:
        report["transfer"][budget] = {}
        for name, actions in arms:
            successful = candidates = operations = 0
            for initial, goal in tasks:
                plan, searched, expanded = synthesize(initial, goal, actions, budget)
                candidates += searched
                operations += expanded
                if plan is not None:
                    assert execute(initial, tuple(op for action in plan for op in action)) == goal
                    successful += 1
            report["transfer"][budget][name] = {
                "passed": successful,
                "candidate_checks": candidates,
                "primitive_checks_charged": operations,
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
