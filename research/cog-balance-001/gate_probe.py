"""Matched-write diagnostic for COG-BALANCE-001 (NumPy, CPU).

Each block has one genuine write for one externally addressed key. Sparse
policies buffer eight events and commit once at the block boundary. Test
queries arrive only after that boundary. Data are synthetic and independent
across calibration and evaluation seeds.

Run: OPENBLAS_NUM_THREADS=1 python research/cog-balance-001/gate_probe.py
"""
import json

import numpy as np


WIDTH = 8
KEYS = 8
VALUES = 4
TEST_SEEDS = (9601, 9611, 9629)
SIGNALS = (0.0, 1.0, 2.0, 4.0)


def panel(rng, worlds, signal):
    shape = (worlds, KEYS, 2, WIDTH)
    event = rng.integers(WIDTH, size=shape[:-1])
    x = rng.normal(size=(*shape, 4))
    target = np.eye(WIDTH, dtype=bool)[event]
    x[..., 0] += signal * target
    return x, target, event


def fit_gate(signal):
    rng = np.random.default_rng(1729)
    x, target, _ = panel(rng, 256, signal)
    design = x.reshape(-1, x.shape[-1])
    # The gate learns a ranking signal from supervised write-event labels.
    return np.linalg.lstsq(design, target.ravel() - 1 / WIDTH, rcond=None)[0]


def take(tokens, indices):
    return np.take_along_axis(tokens, indices[..., None], axis=-1)[..., 0]


def evaluate(seed, signal, weights, worlds=4096):
    rng = np.random.default_rng(seed)
    x, target, event = panel(rng, worlds, signal)
    tokens = rng.integers(VALUES, size=target.shape)
    query = rng.integers(KEYS, size=worlds)
    rows = np.arange(worlds)
    old = take(tokens, event)[rows, query, 1]
    new = (old + rng.integers(1, VALUES, size=worlds)) % VALUES
    revised = tokens.copy()
    revised[rows, query, 1, event[rows, query, 1]] = new
    # Two known operations test downstream sensitivity to the retrieved value.
    # Their modulo composition is supplied exactly in every arm.
    ops = rng.choice((-1, 0, 1), size=(worlds, 2)).sum(axis=1)
    sparse = {
        "learned": (x @ weights).argmax(axis=-1),
        "periodic": np.full(event.shape, WIDTH - 1),
        "random": rng.integers(WIDTH, size=event.shape),
        "oracle": event,
    }
    assert np.array_equal(take(revised, event)[rows, query, 1], new)
    results = {}
    for name, index in sparse.items():
        before = take(tokens, index)[rows, query, 1]
        after = take(revised, index)[rows, query, 1]
        results[name] = metrics(
            before, after, old, new, ops, 1, (index == event).mean(),
        )

    # Dense control sees the same scores and accumulates weighted votes for
    # every candidate in the latest block. Under the generator, the optimal
    # event posterior has logits equal to signal times the shifted feature.
    logits = signal * x[rows, query, 1, :, 0]
    posterior_weights = np.exp(logits - logits.max(axis=1, keepdims=True))

    def posterior_vote(stream):
        votes = np.eye(VALUES)[stream[rows, query, 1]]
        return (votes * posterior_weights[..., None]).sum(axis=1).argmax(axis=1)

    results["dense_signal_oracle"] = metrics(
        posterior_vote(tokens), posterior_vote(revised), old, new, ops, WIDTH, None,
    )

    for alpha in (0.75, 0.97, 0.997, 0.9999):
        # Starting from a zero state, unroll the dense leaky state for the
        # queried key across both blocks and decode the highest value score.
        coeff = (1 - alpha) * alpha ** np.arange(2 * WIDTH - 1, -1, -1)
        eye = np.eye(VALUES)

        def decode(stream):
            values = stream[rows, query].reshape(worlds, 2 * WIDTH)
            return (eye[values] * coeff[None, :, None]).sum(axis=1).argmax(axis=1)

        results[f"dense_{alpha}"] = metrics(
            decode(tokens), decode(revised), old, new, ops, WIDTH, None,
        )
    return results


def metrics(before, after, old, new, ops, writes_per_block, event_accuracy):
    left = (before + ops) % VALUES == (old + ops) % VALUES
    right = (after + ops) % VALUES == (new + ops) % VALUES
    both = left & right
    assert np.array_equal(both, (before == old) & (after == new))
    return {
        "accuracy": float(left.mean()),
        "paired_accuracy": float(both.mean()),
        "event_accuracy": None if event_accuracy is None else float(event_accuracy),
        "writes_per_block": writes_per_block,
    }


def main():
    output = {}
    for signal in SIGNALS:
        weights = fit_gate(signal)
        runs = {seed: evaluate(seed, signal, weights) for seed in TEST_SEEDS}
        for results in runs.values():
            assert results["oracle"]["paired_accuracy"] == 1
            for name in ("learned", "random", "periodic"):
                assert results[name]["writes_per_block"] == 1
        output[signal] = runs
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
