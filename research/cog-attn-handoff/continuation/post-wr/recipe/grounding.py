import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

SEEDS = (13, 37, 61)
SIZES = (32,)
COHORTS = ("familiar", "held_pair", "recombine", "novel_verb", "novel_alias")
MODES = ("weighted16", "analytic")
STEPS, BATCH, TOP, TAIL, TEST = 400, 64, 8, 32, 1024
ROOT = Path(__file__).resolve().parent


def dataset(seed, count, n, aliases, cohort):
    rng = np.random.default_rng(seed)
    text, states, targets, labels = [], [], [], []
    for _ in range(count):
        while True:
            src, dst, decoy = rng.choice(n, 3, replace=False)
            if ((src + dst) % 4 == 0) == (cohort == "held_pair"):
                break
        op, style = rng.integers(2, size=2)
        verb_style = 1 - style if cohort == "recombine" else style
        verb = (("copy", "clone"), ("add", "sum"))[op][verb_style]
        if cohort == "novel_verb":
            verb = ("mirror", "combine")[op]
        src_alias, dst_alias = (str(rng.choice(aliases[key])) for key in (src, dst))
        if cohort == "novel_alias":
            src_alias = f"unknown{src}"
            dst_alias = f"unknown{dst}"
        first, second = (("from", "into"), ("using", "onto"))[style]
        clauses = [f"{first} {src_alias}", f"{second} {dst_alias}"]
        if rng.integers(2):
            clauses.reverse()
        decoy_clause = f"near {rng.choice(aliases[decoy])}"
        clauses.insert(int(rng.integers(3)), decoy_clause)
        text.append(f"Please {verb} {' '.join(clauses)}.")
        state = rng.integers(4, size=n)
        target = state.copy()
        target[dst] = (state[src] + op * state[dst]) % 4
        states.append(state)
        targets.append(target)
        labels.append(op * n * n + src * n + dst)
    return text, np.array(states), np.array(targets), np.array(labels)


def features(text, vocabulary=None):
    rows = []
    for command in text:
        tokens = re.findall(r"[a-z0-9]+", command.lower())
        rows.append(tokens + list(map(" ".join, zip(tokens, tokens[1:]))))
    if vocabulary is None:
        vocabulary = {word: i + 1 for i, word in enumerate(sorted(set(itertools.chain.from_iterable(rows))))}
    return np.array([[vocabulary.get(word, 0) for word in row] for row in rows]), vocabulary


def compatible(actions, state, target):
    n = state.shape[1]
    op, src, dst = actions // (n * n), actions // n % n, actions % n
    row = np.arange(len(state))[:, None]
    changes = state != target
    writes = (state[row, src] + op * state[row, dst]) % 4
    return (writes == target[row, dst]) & (changes.sum(axis=1)[:, None] == changes[row, dst])


def shortlist(probs, n):
    src, dst, op = probs
    s = np.argsort(-src, axis=1, kind="stable")[:, :TOP]
    d = np.argsort(-dst, axis=1, kind="stable")[:, :TOP]
    actions = (np.arange(2)[None, :, None, None] * n * n + s[:, None, :, None] * n
               + d[:, None, None, :]).reshape(len(s), -1)
    mass = action_mass(probs, actions, n)
    index = np.argsort(-mass, axis=1, kind="stable")[:, :TOP]
    return np.take_along_axis(actions, index, axis=1)


def action_mass(probs, actions, n):
    row = np.arange(len(actions))[:, None]
    return (probs[0][row, actions // n % n] * probs[1][row, actions % n]
            * probs[2][row, actions // (n * n)])


def predict(weights, x):
    probs = []
    for weight in weights:
        logits = weight[x].sum(axis=1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs.append(exp / exp.sum(axis=1, keepdims=True))
    return probs


def sample_complement(rng, excluded, universe, budget=TAIL):
    ranks = rng.choice(universe - len(excluded), size=min(budget, universe - len(excluded)), replace=False)
    for action in sorted(excluded):
        ranks += ranks >= action
    return ranks


def residual(rng, excluded, n, changed, budget):
    if len(changed) == 1:
        dst = changed[0]
        blocked = excluded[excluded % n == dst] // n
        draws = sample_complement(rng, blocked, 2 * n, budget) * n + dst
        return draws, 2 * n - len(blocked)
    return sample_complement(rng, excluded, 2 * n * n, budget), 2 * n * n - len(excluded)


def proposals(rng, mode, actions, mask, state, target):
    n = state.shape[1]
    budget = 16 if mode.endswith("16") else TAIL
    rows = np.flatnonzero(~mask.any(axis=1)) if mode == "conditioned" else np.arange(len(mask))
    extended = np.zeros((len(mask), TOP + budget), dtype=int)
    weighted_mask = np.zeros_like(extended, dtype=float)
    extended[:, :TOP], weighted_mask[:, :TOP] = actions, mask
    if not len(rows):
        return extended, weighted_mask, 0
    lengths, multipliers = [], []
    for i in rows:
        draws, size = residual(rng, actions[i], n, np.flatnonzero(state[i] != target[i]), budget)
        extended[i, TOP:TOP + len(draws)] = draws
        lengths.append(len(draws))
        multipliers.append(size / len(draws) if mode.startswith("weighted") else 1.)
    lengths, multipliers = np.array(lengths), np.array(multipliers)
    for length in np.unique(lengths):
        group = lengths == length
        batch_rows = rows[group]
        matching = compatible(extended[batch_rows, TOP:TOP + length], state[batch_rows], target[batch_rows])
        weighted_mask[batch_rows, TOP:TOP + length] = matching * multipliers[group, None]
    return extended, weighted_mask, int(lengths.sum())


def noop_gradient(probs, state):
    src, dst, operation = probs
    groups = state[:, :, None] == np.arange(4)
    src_mass = (src[:, :, None] * groups).sum(axis=1)
    dst_mass = (dst[:, :, None] * groups).sum(axis=1)
    row = np.arange(len(state))[:, None]
    likelihoods = np.column_stack(((src_mass * dst_mass).sum(1), src_mass[:, 0]))
    total = (operation * likelihoods).sum(1, keepdims=True)
    copy, add = operation[:, :1], operation[:, 1:]
    src_given = src * (copy * dst_mass[row, state] + add * (state == 0))
    dst_given = dst * (copy * src_mass[row, state] + add * src_mass[:, :1])
    return [p - contribution / total for p, contribution in
            zip(probs, (src_given, dst_given, operation * likelihoods))]


def train(seed, n, mode, x, states, targets, vocab_size):
    rng = np.random.default_rng(seed)
    exploration = np.random.default_rng(seed + 9000)
    weights = [rng.normal(0, .01, (vocab_size + 1, width)) for width in (n, n, 2)]
    for weight in weights:
        weight[0] = 0
    mean = [np.zeros_like(w) for w in weights]
    variance = [np.zeros_like(w) for w in weights]
    all_actions = np.arange(2 * n * n) if mode == "exhaustive" else None
    stats = dict(examples=STEPS * BATCH, initial_supported=0, supported=0, rescued=0, checks=0)
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        ids = rng.integers(len(x), size=BATCH)
        tokens, state, target = x[ids], states[ids], targets[ids]
        probs = predict(weights, tokens)
        exact = (mode == "analytic") & (state == target).all(axis=1)
        sampled = ~exact
        subset = [p[sampled] for p in probs]
        actions = shortlist(subset, n)
        initial = compatible(actions, state[sampled], target[sampled])
        actions, mask, checks = proposals(exploration, "weighted16", actions, initial,
                                          state[sampled], target[sampled])
        stats["checks"] += checks + initial.size
        stats["initial_supported"] += int(initial.any(1).sum() + exact.sum())
        mass = action_mass(subset, actions, n) * mask
        total = mass.sum(axis=1, keepdims=True)
        valid = total[:, 0] > 0
        stats["rescued"] += int((valid & ~initial.any(1)).sum())
        stats["supported"] += int(valid.sum() + exact.sum())
        posterior = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        exact_delta = noop_gradient([p[exact] for p in probs], state[exact])
        for i, bucket in enumerate((actions // n % n, actions % n, actions // (n * n))):
            approx = subset[i] * valid[:, None]
            np.add.at(approx, (np.arange(len(actions))[:, None], bucket), -posterior)
            delta = np.empty_like(probs[i])
            delta[sampled], delta[exact] = approx, exact_delta[i]
            gradient = np.zeros_like(weights[i])
            np.add.at(gradient, tokens.ravel(), np.repeat(delta / BATCH, tokens.shape[1], axis=0))
            gradient[0] = 0
            mean[i] = .9 * mean[i] + .1 * gradient
            variance[i] = .999 * variance[i] + .001 * gradient ** 2
            weights[i] -= .03 * (mean[i] / (1 - .9 ** step)) / (np.sqrt(variance[i] / (1 - .999 ** step)) + 1e-8)
    stats["seconds"] = time.perf_counter() - start
    return weights, stats


def evaluate(weights, n, x, state, target, gold):
    probs = predict(weights, x)
    candidates = shortlist(probs, n)
    best = probs[2].argmax(1) * n * n + probs[0].argmax(1) * n + probs[1].argmax(1)
    correct = compatible(best[:, None], state, target)[:, 0]
    checks = compatible(candidates, state, target).any(axis=1)
    changed = (state != target).any(axis=1)
    return dict(n=len(x), changed=int(changed.sum()), state_correct=int(correct.sum()),
                changed_state_correct=int((correct & changed).sum()), action_correct=int((best == gold).sum()),
                shortlist_gold=int((candidates == gold[:, None]).any(axis=1).sum()),
                shortlist_compatible=int(checks.sum()), changed_shortlist_compatible=int((checks & changed).sum()))


def main():
    output = ROOT / "evidence"
    output.mkdir(exist_ok=True)
    records = []
    for n in SIZES:
        for seed in SEEDS:
            names = np.array([f"name{i:03d}" for i in range(2 * n)])
            aliases = np.random.default_rng(seed).permutation(names).reshape(n, 2)
            raw, states, targets, _ = dataset(seed + 1000, 4096, n, aliases, "familiar")
            x, vocabulary = features(raw)
            tests = {}
            for offset, cohort in enumerate(COHORTS):
                commands, *data = dataset(seed + 2000 + offset, TEST, n, aliases, cohort)
                tests[cohort] = (features(commands, vocabulary)[0], *data)
            (output / f"data-{n}-{seed}.json").write_text(json.dumps(
                dict(aliases=aliases.tolist(), vocabulary=vocabulary), sort_keys=True) + "\n")
            for mode in MODES:
                weights, stats = train(seed, n, mode, x, states, targets, len(vocabulary))
                scores = {cohort: evaluate(weights, n, *data) for cohort, data in tests.items()}
                records.append(dict(seed=seed, entities=n, mode=mode, training=stats, cohorts=scores))
                np.savez_compressed(output / f"{mode}-{n}-{seed}.npz", *weights)
                (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
                print(n, seed, mode, [v["state_correct"] for v in scores.values()], flush=True)


if __name__ == "__main__":
    main()
