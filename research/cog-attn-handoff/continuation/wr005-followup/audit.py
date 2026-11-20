"""Replay saved evaluation with explicit state mutation and dense top-k."""
import json
from collections import Counter

import numpy as np

from run import BATCH, MODES, ROOT, SEEDS, SIZES, STEPS, TAIL, TOP, dataset, features, predict, sample_complement


def execute(state, action):
    n = len(state)
    op, pair = divmod(int(action), n * n)
    src, dst = divmod(pair, n)
    outcome = state.copy()
    outcome[dst] = (state[src] + op * state[dst]) % 4
    return outcome


def replay(record):
    n, seed, mode = (record[key] for key in ("entities", "seed", "mode"))
    evidence = ROOT / "evidence"
    spec = json.loads((evidence / f"data-{n}-{seed}.json").read_text())
    with np.load(evidence / f"{mode}-{n}-{seed}.npz", allow_pickle=False) as checkpoint:
        weights = [checkpoint[key] for key in ("arr_0", "arr_1", "arr_2")]
    training = record["training"]
    count, initial, supported, rescued, checks = (
        training[key] for key in ("examples", "initial_supported", "supported", "rescued", "checks"))
    assert count == STEPS * BATCH
    assert 0 <= initial <= supported <= count and rescued == supported - initial
    expected_checks = count * (2 * n * n if mode == "exhaustive" else TOP)
    if mode == "exploration":
        expected_checks += TAIL * (count - initial)
    if mode == "conditioned":
        misses = count - initial
        assert expected_checks + min(TAIL, 2 * n - TOP) * misses <= checks
        assert checks <= expected_checks + TAIL * misses
    elif mode.startswith(("unit", "weighted")):
        assert checks == count * (TOP + int(mode[-2:]))
    else:
        assert checks == expected_checks

    for offset, (cohort, expected) in enumerate(record["cohorts"].items()):
        raw, states, targets, gold = dataset(seed + 2000 + offset, expected["n"], n,
                                             np.array(spec["aliases"]), cohort)
        x, _ = features(raw, spec["vocabulary"])
        src, dst, op = predict(weights, x)
        counts = Counter(n=len(raw))
        for i, (state, target, action) in enumerate(zip(states, targets, gold)):
            assert np.array_equal(execute(state, action), target)
            changed = not np.array_equal(state, target)
            # Use the same stable tie convention as the shortlist, including at OOV controls.
            src_order = np.argsort(-src[i], kind="stable")
            dst_order = np.argsort(-dst[i], kind="stable")
            grid = (np.arange(2)[:, None, None] * n * n
                    + src_order[None, :, None] * n + dst_order[None, None, :]).ravel()
            scores = (src[i, grid // n % n] * dst[i, grid % n] * op[i, grid // (n * n)])
            shortlist = grid[np.argsort(-scores, kind="stable")[:TOP]]
            best = op[i].argmax() * n * n + src[i].argmax() * n + dst[i].argmax()
            correct = np.array_equal(execute(state, best), target)
            compatible = any(np.array_equal(execute(state, candidate), target) for candidate in shortlist)
            counts["changed"] += changed
            counts["state_correct"] += correct
            counts["changed_state_correct"] += correct and changed
            counts["action_correct"] += best == action
            counts["shortlist_gold"] += action in shortlist
            counts["shortlist_compatible"] += compatible
            counts["changed_shortlist_compatible"] += changed and compatible
        assert counts == expected, (n, seed, mode, cohort, counts, expected)


if __name__ == "__main__":
    for universe in (128, 2048):
        rng = np.random.default_rng(31)
        for _ in range(100):
            excluded = rng.choice(universe, TOP, replace=False)
            sampled = sample_complement(rng, excluded, universe)
            assert len(set(sampled)) == TAIL
            assert min(sampled) >= 0 and max(sampled) < universe
            assert not set(excluded).intersection(sampled)
    records = json.loads((ROOT / "evidence" / "results.json").read_text())
    assert len(records) == len(MODES) * len(SEEDS) * len(SIZES)
    for row in records:
        replay(row)
        if row["mode"].startswith("weighted"):
            for name, threshold in (("familiar", .95), ("held_pair", .85), ("recombine", .85)):
                cohort = row["cohorts"][name]
                assert cohort["state_correct"] / cohort["n"] >= threshold
                assert cohort["changed_shortlist_compatible"] / cohort["changed"] >= .95
            assert row["training"]["checks"] / row["training"]["examples"] <= 40
    print(f"Verified {len(records)} checkpoints and {len(records) * 5 * 1024:,} evaluations.")
