"""Development-only checks of two acquired procedures on four records."""
import csv
import json
import os
import sys
from itertools import permutations, product
from pathlib import Path

from run import execute, plan, bind
from pilot import learn_trace, primitives


def infer(trace):
    # Infer primitive copy/delete steps from observed states, allowing arbitrary arity.
    body = []
    for before, after in zip(trace, trace[1:]):
        matches = [op for op in primitives(len(before)) if execute(before, (op,)) == after]
        (step,) = matches
        body.append(step)
    return tuple(body)


def controller(initial, goal, program, arity):
    prefix = ()
    state = initial
    if program:
        for operands in permutations(range(len(initial) - 1), arity):
            if any(initial[i] == goal[i] for i in operands):
                continue
            call = bind(program, (*operands, len(initial) - 1))
            candidate = execute(initial, call)
            if all(candidate[i] == goal[i] for i in operands):
                prefix = (call,)
                state = candidate
                break
    actions = (*prefix, *plan(state, goal, None))
    ops = tuple(op for action in actions for op in action)
    assert execute(initial, ops) == goal
    return 4 * len(ops) + len(actions)


def main():
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    swap = learn_trace(((8,3,None),(8,3,8),(3,3,8),(3,8,8),(3,8,None)))
    rotate = infer(((8,3,5,None),(8,3,5,8),(3,3,5,8),(3,5,5,8),(3,5,8,8),(3,5,8,None)))
    for a,b,c in product(range(3), repeat=3):
        assert execute((a,b,c,None),rotate) == (b,c,a,None)
        assert execute((a,b,None),swap) == (b,a,None)
    totals = [0]*4
    with (output/"panel.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["goal","primitive","swap","rotate","hindsight"])
        for target in product(range(4),repeat=4):
            costs = [controller((0,1,2,3,None), (*target,None), program, arity)
                     for program,arity in (((),0),(swap,2),(rotate,3))]
            row = [*costs,min(costs)]
            totals = [a+b for a,b in zip(totals,row)]
            writer.writerow(["".join(map(str,target)),*row])
    (output/"metrics.json").write_text(json.dumps(dict(zip(
        ("primitive","swap","rotate","hindsight"), totals))))
    (output/"programs.json").write_text(json.dumps({"swap":swap,"rotate":rotate}))


if __name__ == "__main__":
    main()
