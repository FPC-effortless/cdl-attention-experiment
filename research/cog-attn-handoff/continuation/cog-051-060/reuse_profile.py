"""Learn eight frequent goals per observed epoch and bound compilation churn."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from panel import Panel, plan, probe, rollout
from policy_repair import reverse_policy


def run(seed, hot):
    rng = np.random.default_rng(seed)
    clock = perf_counter_ns
    totals = Counter()
    for turn in range(1536):
        if turn % 512 == 0:
            env = Panel(rng)
            model = probe(env, rng, True)
            counts, policies = Counter(), {}
        start = int(rng.integers(32))
        goal = int(rng.integers(4 if hot and rng.random() < .85 else 32))
        def baseline():
            return plan(model, start, goal)
        def compiled():
            if turn % 512 < 64:
                counts[goal] += 1
            if turn % 512 == 63:
                policies.update({g: reverse_policy(model, g) for g, _ in counts.most_common(8)})
            if goal in policies:
                return policies[goal][start]
            return baseline()
        methods = [('baseline', baseline), ('compiled', compiled)]
        if rng.integers(2):
            methods.reverse()
        answers = {}
        for label, method in methods:
            begin = clock()
            answers[label] = method()
            totals[label + '_ns'] += clock() - begin
        assert len(answers['compiled']) == len(answers['baseline'])
        assert rollout(env, start, answers['compiled']) == goal
        totals['correct'] += 1
        totals['hits'] += goal in policies
    return totals


if __name__ == '__main__':
    output = {seed: {arm: run(seed, arm == 'hot') for arm in ('uniform', 'hot')}
              for seed in (55103, 55109, 55117, 55127, 55147)}
    (Path(__file__).resolve().parent/'results/profile-reuse.json').write_text(json.dumps(output))
