#!/usr/bin/env python3
"""Deterministic numerical checks for CSM-MATH-001's mathematical review.

These are synthetic algebra checks and bounded counterexamples, not trained
model evaluations. The analytic assumptions and proofs are documented in
independent_review.txt. Only Python's standard library and NumPy are used.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np


SEED = 20261109


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def central_derivative(fn, x: float, step: float = 1e-6) -> float:
    return (fn(x + step) - fn(x - step)) / (2.0 * step)


def central_jacobian(fn, x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    columns = []
    for index in range(x.size):
        direction = np.zeros_like(x, dtype=float)
        direction[index] = step
        columns.append((fn(x + direction) - fn(x - direction)) / (2 * step))
    return np.stack(columns, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "verification" / "stability.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    checks: list[dict] = []

    def record(name: str, passed: bool, **details) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    # Bounded regulator and controller in the narrow token-only candidate.
    # The synthetic weights below are not weights from any CSM checkpoint.
    lambdas = np.array([2 ** (-1 / 2)] * 6 + [2 ** (-1 / 32)] * 6)
    matrix_b = rng.normal(size=(12, 7))
    bias_b = rng.normal(size=12)
    matrix_c = rng.normal(size=(9, 7 + 4))
    matrix_ur = rng.normal(size=(9, 12))
    trace = rng.uniform(-1, 1, size=12)
    max_abs_trace = float(np.max(np.abs(trace)))
    max_abs_controller = 0.0
    for _ in range(2048):
        features = rng.normal(size=7)
        proposal = np.tanh(matrix_b @ features + bias_b)
        trace = lambdas * trace + (1 - lambdas) * proposal
        cursor_record = rng.uniform(-1, 1, size=4)
        controller = np.tanh(
            matrix_c @ np.concatenate((features, cursor_record)) + matrix_ur @ trace
        )
        max_abs_trace = max(max_abs_trace, float(np.max(np.abs(trace))))
        max_abs_controller = max(
            max_abs_controller, float(np.max(np.abs(controller)))
        )
    record(
        "token_only_regulator_and_controller_coordinate_bounds",
        max_abs_trace <= 1 + 1e-12 and max_abs_controller <= 1 + 1e-12,
        synthetic_steps=2048,
        regulator_dimensions=12,
        lambda_values=[float(lambdas[0]), float(lambdas[-1])],
        max_absolute_regulator=max_abs_trace,
        max_absolute_controller=max_abs_controller,
        scope="Numerical corroboration of convex/tanh bounds; no contraction claim for controller.",
    )

    first = rng.uniform(-1, 1, size=12)
    second = rng.uniform(-1, 1, size=12)
    initial_delta = first - second
    for _ in range(128):
        common_h = rng.uniform(-1, 1, size=12)
        first = lambdas * first + (1 - lambdas) * common_h
        second = lambdas * second + (1 - lambdas) * common_h
    expected_delta = lambdas**128 * initial_delta
    delta_error = float(np.max(np.abs(first - second - expected_delta)))
    record(
        "exogenous_trace_exact_initial_condition_forgetting",
        delta_error < 1e-12,
        steps=128,
        max_absolute_identity_error=delta_error,
        observed_delta_max=float(np.max(np.abs(first - second))),
        predicted_delta_max=float(np.max(np.abs(expected_delta))),
        scope="Identical token inputs; regulator only.",
    )

    original = rng.uniform(-3, 3, size=(6000, 8, 4))
    targets = rng.uniform(-3, 3, size=(6000, 4))
    destinations = rng.integers(0, 8, size=6000)
    gains = rng.uniform(0, 1, size=6000)
    gains[0:2] = [0, 1]  # Include the closed-interval boundary cases.
    updated = original.copy()
    batch_indices = np.arange(original.shape[0])
    old_selected = original[batch_indices, destinations]
    updated[batch_indices, destinations] = (
        (1 - gains[:, None]) * old_selected + gains[:, None] * targets
    )
    untouched = np.ones(original.shape[:2], dtype=bool)
    untouched[batch_indices, destinations] = False
    inactive_error = float(np.max(np.abs(updated[untouched] - original[untouched])))
    record(
        "convex_value_bound_and_protected_rows",
        float(np.max(np.abs(updated))) <= 3 + 1e-12 and inactive_error == 0,
        synthetic_updates=6000,
        coordinate_radius=3,
        max_absolute_value=float(np.max(np.abs(updated))),
        inactive_row_max_change=inactive_error,
        scope="Conditional on bounded proposals and gains in [0,1].",
    )

    gate_map = lambda v: (1 - sigmoid(8 * v)) * v + sigmoid(8 * v)
    analytic_gate_derivative = 0.5 + 8 / 4
    numerical_gate_derivative = central_derivative(gate_map, 0)
    gate_distance_ratio = (gate_map(0.01) - gate_map(0)) / 0.01
    grid = np.linspace(-1, 1, 4001)
    gate_outputs = np.array([gate_map(float(v)) for v in grid])
    record(
        "state_dependent_gate_bounded_noncontractive_counterexample",
        abs(numerical_gate_derivative - analytic_gate_derivative) < 1e-8
        and gate_distance_ratio > 1
        and float(np.max(np.abs(gate_outputs))) <= 1 + 1e-12,
        analytic_derivative_at_zero=analytic_gate_derivative,
        finite_difference_derivative_at_zero=numerical_gate_derivative,
        incorrectly_dropped_gate_term_derivative=0.5,
        input_pair=[0, 0.01],
        output_pair=[gate_map(0), gate_map(0.01)],
        distance_amplification=gate_distance_ratio,
        sampled_output_min=float(gate_outputs.min()),
        sampled_output_max=float(gate_outputs.max()),
        scope="Disproves a general contraction claim; not a trained-model instability measurement.",
    )

    repelling_map = lambda v: 0.5 * v + 0.5 * math.tanh(3 * v)
    repelling_derivative = central_derivative(repelling_map, 0)
    repelling_outputs = np.array([repelling_map(float(v)) for v in grid])
    record(
        "bounded_proposal_repelling_equilibrium_counterexample",
        repelling_map(0) == 0
        and abs(repelling_derivative - 2) < 1e-8
        and float(np.max(np.abs(repelling_outputs))) <= 1 + 1e-12,
        analytic_derivative_at_zero=2.0,
        finite_difference_derivative_at_zero=repelling_derivative,
        sampled_output_min=float(repelling_outputs.min()),
        sampled_output_max=float(repelling_outputs.max()),
        scope="Bounded nonlinear proposal family only; explicitly not an unbounded recurrence or actual-model result.",
    )

    # Check the full matrix Jacobian, including the rank-one gate term.
    value = np.array([0.12, -0.17])
    proposal_matrix = np.array([[0.2, -0.3], [0.1, 0.25]])
    proposal_bias = np.array([0.05, -0.02])
    gate_vector = np.array([1.1, -0.7])
    epsilon = 0.8

    def vector_write(v):
        z = np.tanh(proposal_matrix @ v + proposal_bias)
        a = epsilon * sigmoid(float(gate_vector @ v))
        return (1 - a) * v + a * z

    z = np.tanh(proposal_matrix @ value + proposal_bias)
    s = sigmoid(float(gate_vector @ value))
    a = epsilon * s
    derivative_z = np.diag(1 - z**2) @ proposal_matrix
    gradient_a = epsilon * s * (1 - s) * gate_vector
    analytic_jacobian = (
        (1 - a) * np.eye(2) + a * derivative_z + np.outer(z - value, gradient_a)
    )
    numeric_jacobian = central_jacobian(vector_write, value)
    jacobian_error = float(np.max(np.abs(analytic_jacobian - numeric_jacobian)))
    record(
        "complete_gate_and_proposal_jacobian_formula",
        jacobian_error < 1e-8,
        analytic_jacobian=analytic_jacobian.tolist(),
        finite_difference_jacobian=numeric_jacobian.tolist(),
        max_absolute_error=jacobian_error,
    )

    # A constructed active-record subsystem satisfying the sufficient bound:
    # a=.5, z=.2*tanh(w)+.1*tanh(r_next), rho=.9, common h=.3.
    # Thus q_w=.6, K_r=.05; Q=.95 and c=1 satisfy the weighted-norm condition.
    rho, q_w, k_r, q_total, weight_c = 0.9, 0.6, 0.05, 0.95, 1.0

    def stable_subsystem(state):
        w, r = state
        r_next = rho * r + (1 - rho) * 0.3
        z_next = 0.2 * math.tanh(w) + 0.1 * math.tanh(r_next)
        return np.array([0.5 * w + 0.5 * z_next, r_next])

    ratios = []
    for _ in range(6000):
        pair = rng.uniform(-1, 1, size=(2, 2))
        initial_distance = abs(pair[0, 0] - pair[1, 0]) + weight_c * abs(
            pair[0, 1] - pair[1, 1]
        )
        if initial_distance == 0:
            continue
        left, right = stable_subsystem(pair[0]), stable_subsystem(pair[1])
        final_distance = abs(left[0] - right[0]) + weight_c * abs(left[1] - right[1])
        ratios.append(final_distance / initial_distance)
    sufficient_c = k_r * rho / (q_total - rho)
    record(
        "sufficient_weighted_norm_stability_constructed_subsystem",
        max(q_w, rho) < q_total < 1
        and weight_c + 1e-12 >= sufficient_c
        and max(ratios) <= q_total + 1e-12,
        q_w=q_w,
        rho=rho,
        regulator_to_write_lipschitz=k_r,
        chosen_Q=q_total,
        chosen_c=weight_c,
        sufficient_minimum_c=sufficient_c,
        maximum_sampled_weighted_distance_ratio=max(ratios),
        synthetic_pairs=len(ratios),
        scope="An explicit subsystem satisfying assumptions; does not certify CSM parameter norms.",
    )

    unchanged_slot_map = lambda v: np.array([0.5 * v[0] + 0.5 * v[1], v[1]])
    unchanged_slot_jacobian = central_jacobian(unchanged_slot_map, np.array([0.1, 0.2]))
    eigenvalues = np.linalg.eigvals(unchanged_slot_jacobian)
    record(
        "untouched_record_prevents_strict_global_contraction",
        float(np.min(np.abs(eigenvalues - 1))) < 1e-8,
        jacobian=unchanged_slot_jacobian.tolist(),
        eigenvalues=[float(value.real) for value in eigenvalues],
        spectral_norm=float(np.linalg.norm(unchanged_slot_jacobian, ord=2)),
        scope="Identity record coordinate gives an eigenvalue of one.",
    )

    def hard_route(values):
        result = values.copy()
        destination = 0 if values[0] >= 0 else 1
        result[destination] = 0.5 * values[destination] + 0.5
        return result

    route_left, route_right = np.array([0.001, 0.0]), np.array([-0.001, 0.0])
    route_out_left, route_out_right = hard_route(route_left), hard_route(route_right)
    input_distance = float(np.max(np.abs(route_left - route_right)))
    output_distance = float(np.max(np.abs(route_out_left - route_out_right)))
    record(
        "hard_route_boundary_bounded_discontinuity_counterexample",
        abs(input_distance - 0.002) < 1e-12
        and abs(output_distance - 0.5015) < 1e-12,
        input_pair=[route_left.tolist(), route_right.tolist()],
        output_pair=[route_out_left.tolist(), route_out_right.tolist()],
        input_max_norm_distance=input_distance,
        output_max_norm_distance=output_distance,
        distance_amplification=output_distance / input_distance,
        scope="Crosses a decision boundary; does not contradict fixed-route local bounds.",
    )

    logits_matrix = np.array([[2.0, 0.0], [-1.0, 0.0], [0.0, 0.5]])
    routing_point = np.array([1.0, 0.2])
    logits = logits_matrix @ routing_point
    winner = int(np.argmax(logits))
    margin = float(logits[winner] - np.max(np.delete(logits, winner)))
    lipschitz = float(np.max(np.linalg.norm(logits_matrix, axis=1)))
    radius = margin / (2 * lipschitz)
    preserved = True
    max_perturbation = 0.0
    for _ in range(6000):
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        delta = direction * radius * 0.99 * rng.random()
        max_perturbation = max(max_perturbation, float(np.linalg.norm(delta)))
        preserved = preserved and int(np.argmax(logits_matrix @ (routing_point + delta))) == winner
    record(
        "argmax_margin_sufficient_radius",
        preserved and max_perturbation < radius,
        logits=logits.tolist(),
        winner=winner,
        margin=margin,
        per_logit_euclidean_lipschitz=lipschitz,
        sufficient_strict_radius=radius,
        maximum_sampled_perturbation=max_perturbation,
        synthetic_perturbations=6000,
        scope="Numerical corroboration of the analytic one-step margin theorem.",
    )

    def write_a(v):
        result = v.copy()
        result[0] = 0.5 * v[0] + 0.5 * v[1]
        return result

    def write_b(v):
        result = v.copy()
        result[1] = 0.5 * v[1] + 0.5 * v[0]
        return result

    initial = np.array([0.0, 1.0])
    a_then_b = write_b(write_a(initial))
    b_then_a = write_a(write_b(initial))
    simultaneous = np.array([0.5, 0.5])
    record(
        "disjoint_write_destinations_with_read_dependencies_do_not_commute",
        np.allclose(a_then_b, [0.5, 0.75], atol=0, rtol=0)
        and np.allclose(b_then_a, [0.25, 0.5], atol=0, rtol=0)
        and not np.array_equal(a_then_b, b_then_a),
        initial=initial.tolist(),
        a_then_b=a_then_b.tolist(),
        b_then_a=b_then_a.tolist(),
        original_snapshot_simultaneous=simultaneous.tolist(),
    )

    gains_a, gains_b, target_one, target_two = 0.3, 0.7, 0.8, -0.4
    write_one = lambda v: (1 - gains_a) * v + gains_a * target_one
    write_two = lambda v: (1 - gains_b) * v + gains_b * target_two
    order_difference = write_two(write_one(0.2)) - write_one(write_two(0.2))
    expected_difference = gains_a * gains_b * (target_two - target_one)
    record(
        "same_destination_affine_write_order_formula",
        abs(order_difference - expected_difference) < 1e-12,
        observed_difference=order_difference,
        analytic_difference=expected_difference,
    )

    v, target, gain = 0.0, 1.0, sigmoid(0)
    first_write = (1 - gain) * v + gain * target
    for _ in range(20):
        v = (1 - gain) * v + gain * target
    expected_residual = (1 - gain) ** 20
    record(
        "finite_sigmoid_write_residual_is_not_exact_vector_set",
        first_write != target and abs((target - v) - expected_residual) < 1e-12,
        gain=gain,
        first_write_value=first_write,
        target=target,
        writes=20,
        observed_residual=target - v,
        predicted_residual=expected_residual,
        scope="Vector semantics only; does not imply a discrete-class recall error.",
    )

    slow_lambda = 2 ** (-1 / 32)
    cold_trace = 0.0
    first_trace = 1 - slow_lambda
    for _ in range(32):
        cold_trace = slow_lambda * cold_trace + (1 - slow_lambda)
    record(
        "token_half_life_and_declared_zero_initialization",
        abs(cold_trace - 0.5) < 1e-12,
        half_life_tokens=32,
        lambda_value=slow_lambda,
        constant_input=1,
        initial_trace=0,
        trace_after_one_token=first_trace,
        trace_after_32_tokens=cold_trace,
        scope="Cold-start trace is not an unbiased stationary-mean estimator.",
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "schema": "csm-math-001-independent-stability-v1",
        "status": "passed" if passed else "failed",
        "random_seed": SEED,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "scope": "Deterministic synthetic algebra verification and bounded counterexamples, not trained-model evaluation.",
        "excluded_claims": [
            "No training or language-quality result.",
            "No speed benchmark or whole-sequence parallelism guarantee.",
            "No global CSM contraction or stability certificate.",
            "No unbounded linear recurrence used as evidence against the bounded-proposal model.",
            "Numerical tests corroborate; they do not replace analytic proofs.",
        ],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed_count"], "total": len(checks), "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
