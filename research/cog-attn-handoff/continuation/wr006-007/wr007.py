"""Calibrated sequential promotion on simulated meaning verification."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

BUDGET = 7
VERIFY, DEFER, RISK_LIMIT = .15, 1., .20
COSTS = (20., 100.)
MODES = ("original", "pooled", "critical", "adaptive_pooled", "adaptive_context")


def posterior(evidence, noise):
    return 1. / (1. + np.exp(-evidence * np.log((1. - noise) / noise)))


@lru_cache(maxsize=None)
def decision(mode, noise, cost, used, evidence):
    prob = posterior(evidence, noise)
    risk = cost * min(prob, 1. - prob)
    if mode in ("original", "pooled", "critical"):
        eligible = min(prob, 1. - prob) <= .05 if mode != "critical" else risk <= RISK_LIMIT
        if used and eligible:
            return "promote", risk
        return ("verify", 0.) if used < BUDGET else ("defer", DEFER)

    options = [(DEFER, "defer")]
    if used and risk <= RISK_LIMIT:
        options.append((risk, "promote"))
    if used < BUDGET:
        positive = prob * (1. - noise) + (1. - prob) * noise
        high = decision(mode, noise, cost, used + 1, evidence + 1)[1]
        low = decision(mode, noise, cost, used + 1, evidence - 1)[1]
        options.append((VERIFY + positive * high + (1. - positive) * low, "verify"))
    value, choice = min(options)
    return choice, value


def evaluate(observations, truth, mode, noise, cost):
    wrong = promoted = acquired = 0
    for readings, meaning in zip(observations, truth):
        evidence, used = 0, 0
        choice = decision(mode, noise, cost, used, evidence)[0]
        while choice == "verify":
            evidence += 1 if readings[used] else -1
            used += 1
            choice = decision(mode, noise, cost, used, evidence)[0]
        acquired += used
        if choice == "promote":
            promoted += 1
            wrong += (evidence > 0) != meaning
    total = len(truth)
    return dict(total=total, promoted=int(promoted), wrong=int(wrong), observations=acquired,
                realized_cost=VERIFY * acquired + DEFER * (total - promoted) + cost * wrong)


def main():
    rates = (.05, .15)
    calibration = np.random.default_rng(1701)
    errors = (calibration.random((2, 8192)) < np.array(rates)[:, None]).sum(axis=1)
    noise = (errors + 1) / 8194
    records = []
    for seed in range(1720, 1728):
        rng = np.random.default_rng(seed)
        for context, rate in enumerate(rates):
            truth = rng.integers(2, size=1024).astype(bool)
            for regime, width in (("independent", BUDGET), ("correlated", 1)):
                flips = rng.random((len(truth), width)) < rate
                observations = np.broadcast_to(truth[:, None] ^ flips, (len(truth), BUDGET))
                for criticality, cost in enumerate(COSTS):
                    for mode in MODES:
                        assumed = (.10 if mode == "original" else
                                   float(noise[context]) if mode == "adaptive_context" else float(noise.mean()))
                        stats = evaluate(observations, truth, mode, assumed, cost)
                        assert 0 <= stats["wrong"] <= stats["promoted"] <= stats["total"]
                        assert stats["observations"] <= BUDGET * stats["total"]
                        records.append(dict(seed=seed, context=context, rate=rate, regime=regime,
                                            criticality=criticality, mode=mode, assumed=assumed, **stats))
    payload = dict(calibration_seed=1701, calibration_errors=errors.tolist(),
                   calibration_trials_per_context=8192, records=records)
    Path(__file__).with_name("wr007.json").write_text(json.dumps(payload, indent=2) + "\n")
    for regime in ("independent", "correlated"):
        for criticality in (0, 1):
            for mode in MODES:
                group = [r for r in records if r["regime"] == regime and r["criticality"] == criticality
                         and r["mode"] == mode]
                totals = {key: sum(r[key] for r in group)
                          for key in ("total", "wrong", "promoted", "observations", "realized_cost")}
                print(regime, criticality, mode, totals)


if __name__ == "__main__":
    main()
