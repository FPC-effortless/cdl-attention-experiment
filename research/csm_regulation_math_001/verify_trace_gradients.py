#!/usr/bin/env python3
"""Finite-difference check of the smooth trace/controller subgraph only.

No derivative of argmax, hard routing, selection, or the full CSM is claimed.
Run: python verify_trace_gradients.py
Writes verification/trace_gradients.json beside this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def forward(p, fixed):
    state = p["initial"].copy()
    records = []
    loss = 0.0
    for t in range(len(fixed["features"])):
        if fixed["reset"][t] and not fixed["valid"][t]:
            raise ValueError("Reset must coincide with a valid token")
        if not fixed["valid"][t]:
            records.append(None)
            continue
        if fixed["reset"][t]:
            state = np.zeros_like(state)
        feature = fixed["features"][t]
        signal = np.tanh(p["B"] @ feature + p["b"])
        state = fixed["decay"] * state + (1.0 - fixed["decay"]) * signal
        output = np.tanh(fixed["A"] @ feature + p["U"] @ state)
        error = output - fixed["targets"][t]
        loss += 0.5 * float(error @ error)
        records.append((feature, signal, state.copy(), output, error))
    return loss, records


def reverse_mode(p, fixed):
    loss, records = forward(p, fixed)
    grads = {key: np.zeros_like(value) for key, value in p.items()}
    carry = np.zeros_like(p["initial"])
    lam = fixed["decay"]
    for t in range(len(records) - 1, -1, -1):
        if records[t] is None:
            # Padding has no loss and an identity state transition.
            continue
        feature, signal, state, output, error = records[t]
        bar_precontroller = error * (1.0 - output * output)
        grads["U"] += np.outer(bar_precontroller, state)
        bar_state = carry + p["U"].T @ bar_precontroller
        bar_predriver = bar_state * (1.0 - lam) * (1.0 - signal * signal)
        grads["B"] += np.outer(bar_predriver, feature)
        grads["b"] += bar_predriver
        carry = np.zeros_like(carry) if fixed["reset"][t] else lam * bar_state
    grads["initial"] = carry
    return loss, grads


def central_differences(p, fixed, epsilon):
    numerical = {}
    for name, value in p.items():
        derivative = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            old = float(value[index])
            value[index] = old + epsilon
            plus = forward(p, fixed)[0]
            value[index] = old - epsilon
            minus = forward(p, fixed)[0]
            value[index] = old
            derivative[index] = (plus - minus) / (2.0 * epsilon)
        numerical[name] = derivative
    return numerical


def check_case(name, p, fixed, epsilon=1e-6, atol=5e-8, rtol=1e-6):
    loss, analytic = reverse_mode(p, fixed)
    numeric = central_differences(p, fixed, epsilon)
    summaries = {}
    for key in p:
        difference = np.abs(analytic[key] - numeric[key])
        scaled = difference / (1.0 + np.abs(analytic[key]) + np.abs(numeric[key]))
        passed = bool(np.allclose(analytic[key], numeric[key], atol=atol, rtol=rtol))
        summaries[key] = {
            "shape": list(p[key].shape),
            "passed": passed,
            "max_absolute_error": float(difference.max()),
            "max_scaled_error": float(scaled.max()),
            "analytic_l2_norm": float(np.linalg.norm(analytic[key])),
            "numerical_l2_norm": float(np.linalg.norm(numeric[key])),
        }
    leading_reset = bool(fixed["reset"][np.flatnonzero(fixed["valid"])[0]])
    initial_is_exactly_zero = bool(np.array_equal(analytic["initial"],
                                                np.zeros_like(analytic["initial"])))
    if leading_reset:
        assert initial_is_exactly_zero, "Reset must cut the initial-state gradient"
    return {
        "name": name,
        "passed": all(item["passed"] for item in summaries.values()),
        "loss": loss,
        "valid": fixed["valid"].tolist(),
        "reset_before_valid_token": fixed["reset"].tolist(),
        "decays": fixed["decay"].tolist(),
        "leading_reset": leading_reset,
        "initial_gradient_exactly_zero": initial_is_exactly_zero,
        "parameters": summaries,
    }


def review_reference_counts():
    # Importing the local reference checks instantiated arrays, not just formulas.
    from reference import Config, counts, parameters

    base = Config(traces=0)
    regulated = Config(traces=12)
    base_p = parameters(base, seed=20261109)
    regulated_p = parameters(regulated, seed=20261109)
    base_count = sum(value.size for value in base_p.values())
    regulated_count = sum(value.size for value in regulated_p.values())
    seed_match = all(np.array_equal(value, regulated_p[key])
                     for key, value in base_p.items())
    declared = counts(regulated)
    passed = (base_count == declared["base_parameters"] == 18294
              and regulated_count == declared["candidate_parameters"] == 19074
              and seed_match)
    return {
        "passed": bool(passed),
        "instantiated_base_parameters": int(base_count),
        "instantiated_regulated_parameters": int(regulated_count),
        "seeded_base_parameters_equal": bool(seed_match),
        "declared_dense_nominal_macs": declared["candidate_dense_nominal_macs"],
        "declared_dense_cache_fp32_bytes": declared["candidate_dense_cache_fp32_bytes"],
        "scope": "Parameter arrays independently counted; dense MAC/cache formulas are declarations, not measured indexed runtime costs.",
    }


def main():
    rng = np.random.default_rng(20261109)
    n, d, m = 7, 3, 2
    params = {
        "B": rng.normal(0.0, 0.3, (m, d)),
        "b": rng.normal(0.0, 0.1, m),
        "U": rng.normal(0.0, 0.3, (d, m)),
        "initial": rng.normal(0.0, 0.2, m),
    }
    fixed_base = {
        "A": rng.normal(0.0, 0.2, (d, d)),
        "features": rng.normal(0.0, 0.6, (n, d)),
        "targets": rng.normal(0.0, 0.4, (n, d)),
        "decay": np.exp(-1.0 / np.array([4.0, 32.0])),
    }
    specifications = [
        ("continuous_sequence", [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], None),
        ("padding_and_midsequence_reset", [1, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0], None),
        ("leading_reset_and_padding", [0, 1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0], None),
        ("immediate_trace_with_reset", [1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0], np.zeros(m)),
    ]
    cases = []
    for name, valid, reset, decay_override in specifications:
        fixed = dict(fixed_base)
        fixed["valid"] = np.array(valid, dtype=bool)
        fixed["reset"] = np.array(reset, dtype=bool)
        if decay_override is not None:
            fixed["decay"] = decay_override
        cases.append(check_case(name, params, fixed))
    reference_review = review_reference_counts()
    result = {
        "status": "passed" if all(case["passed"] for case in cases) and reference_review["passed"] else "failed",
        "scope": "Differentiable regulator and tanh controller path only; no hard-routing or full-model backward claim.",
        "dtype": "numpy.float64",
        "numpy_version": np.__version__,
        "seed": 20261109,
        "central_difference_epsilon": 1e-6,
        "absolute_tolerance": 5e-8,
        "relative_tolerance": 1e-6,
        "loss": "0.5 * sum over valid tokens of ||u_t - target_t||^2",
        "equations": [
            "s_t = tanh(B f_t + b)",
            "r_t = lambda*r_previous + (1-lambda)*s_t",
            "u_t = tanh(A f_t + U r_t)",
            "bar_z_t = (u_t-target_t)*(1-u_t^2)",
            "bar_r_t = future_carry + U^T bar_z_t",
            "bar_driver_t = (1-s_t^2)*(1-lambda)*bar_r_t",
            "grad_B += outer(bar_driver_t, f_t); grad_b += bar_driver_t",
            "grad_U += outer(bar_z_t, r_t)",
            "future_carry = lambda*bar_r_t unless reset, where it is zero; padding leaves carry unchanged",
        ],
        "cases": cases,
        "reference_review": reference_review,
        "limitations": [
            "Finite-difference agreement at these finite float64 cases is not a formal all-input proof.",
            "Lambda is fixed; no derivative with respect to decay parameters is checked.",
            "Feature inputs and A are held fixed; the recurrent main CSM is outside this check.",
            "No training result, FP32 backward equivalence, surrogate estimator, throughput or quality claim is established.",
        ],
    }
    path = Path(__file__).resolve().parent / "verification" / "trace_gradients.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    maximum = max(p["max_absolute_error"] for case in cases for p in case["parameters"].values())
    print(json.dumps({"status": result["status"], "cases": len(cases),
                      "parameter_coordinates_per_case": sum(a.size for a in params.values()),
                      "max_absolute_gradient_error": maximum,
                      "reference_parameter_review": reference_review,
                      "output": str(path)}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
