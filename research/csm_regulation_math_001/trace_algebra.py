#!/usr/bin/env python3
"""Exact finite checks of diagonal affine trace algebra (standard library only).

This is an executable checker, not a formal proof. Fraction arithmetic isolates
the algebra from floating-point reassociation. Candidate exp(-1/tau) coefficients
are checked as exact rational representations of their binary64 approximations;
those cases do not claim exact transcendental coefficients or FP32 equivalence.

Run: python trace_algebra.py
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import json
import math
import random


Vector = tuple[Q, ...]


@dataclass(frozen=True)
class Affine:
    a: Vector
    b: Vector

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b):
            raise ValueError("Affine coefficient widths differ")


@dataclass(frozen=True)
class Event:
    signal: Vector
    valid: bool = True
    reset: bool = False


def identity(width: int) -> Affine:
    return Affine((Q(1),) * width, (Q(0),) * width)


def compose(first: Affine, second: Affine) -> Affine:
    """Chronological composition: apply first, then second."""
    if len(first.a) != len(second.a):
        raise ValueError("Composition widths differ")
    return Affine(
        tuple(a2 * a1 for a1, a2 in zip(first.a, second.a)),
        tuple(a2 * b1 + b2 for b1, a2, b2 in zip(first.b, second.a, second.b)),
    )


def apply(transform: Affine, state: Vector) -> Vector:
    if len(transform.a) != len(state):
        raise ValueError("State and transform widths differ")
    return tuple(a * r + b for a, r, b in zip(transform.a, state, transform.b))


def event_transform(decays: Vector, event: Event, reset_state: Vector) -> Affine:
    """Pad is identity; reset occurs before the valid token's ordinary update."""
    width = len(decays)
    if len(event.signal) != width or len(reset_state) != width:
        raise ValueError("Event widths differ")
    if any(lam < 0 or lam > 1 for lam in decays):
        raise ValueError("Convex trace requires decay coefficients in [0, 1]")
    if event.reset and not event.valid:
        raise ValueError("A reset must coincide with an observed token")
    if not event.valid:
        return identity(width)
    if event.reset:
        return Affine(
            (Q(0),) * width,
            tuple(lam * init + (1 - lam) * signal
                  for lam, init, signal in zip(decays, reset_state, event.signal)),
        )
    return Affine(
        decays,
        tuple((1 - lam) * signal for lam, signal in zip(decays, event.signal)),
    )


def reference_serial(decays: Vector, events: list[Event], initial: Vector,
                     reset_state: Vector) -> list[Vector]:
    """Direct recurrence, independent of the affine event construction."""
    state = initial
    states: list[Vector] = []
    for event in events:
        if event.reset and not event.valid:
            raise ValueError("A reset must coincide with an observed token")
        if event.valid:
            if event.reset:
                state = reset_state
            state = tuple(lam * old + (1 - lam) * signal
                          for lam, old, signal in zip(decays, state, event.signal))
        states.append(state)
    return states


def balanced_inclusive_scan(transforms: list[Affine], width: int) -> list[Affine]:
    """Work-efficient tree scan, executed serially here for exact checking.

    Upsweep/down-sweep have logarithmic parallel span. With P the next power
    of two >= N, this uses 2(P-1)+N compositions, so O(N*width) arithmetic
    operations under the unit-cost arithmetic model. Exact Fraction bit costs
    grow with sequence length and are not a production performance benchmark.
    """
    n = len(transforms)
    if n == 0:
        return []
    size = 1 << (n - 1).bit_length()
    unit = identity(width)
    totals = [unit] * (2 * size)
    totals[size:size + n] = transforms
    for node in range(size - 1, 0, -1):
        totals[node] = compose(totals[2 * node], totals[2 * node + 1])
    prefixes = [unit] * (2 * size)
    for node in range(1, size):
        prefixes[2 * node] = prefixes[node]
        prefixes[2 * node + 1] = compose(prefixes[node], totals[2 * node])
    return [compose(prefixes[size + i], transforms[i]) for i in range(n)]


def check_associativity() -> int:
    pairs = [Affine((a,), (b,))
             for a in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1))
             for b in (Q(-1), Q(0), Q(1))]
    count = 0
    unit = identity(1)
    for p in pairs:
        assert compose(unit, p) == p == compose(p, unit)
        for q in pairs:
            for r in pairs:
                assert compose(compose(p, q), r) == compose(p, compose(q, r))
                count += 1
    p = Affine((Q(1, 2),), (Q(1),))
    q = Affine((Q(1, 3),), (Q(2),))
    assert compose(p, q) != compose(q, p), "Chronological order must matter"
    return count


def check_closed_form() -> int:
    count = 0
    for lam in (Q(0), Q(1, 4), Q(1, 2), Q(15, 16), Q(1)):
        initial = (Q(2, 3),)
        inputs = [Q((i % 5) - 2, 3) for i in range(12)]
        states = reference_serial((lam,), [Event((u,)) for u in inputs], initial, (Q(0),))
        for t, state in enumerate(states, 1):
            expected = lam ** t * initial[0] + (1 - lam) * sum(
                (lam ** (t - k)) * inputs[k - 1] for k in range(1, t + 1)
            )
            assert state == (expected,)
            count += 1
        impulse = [Event((Q(1),))] + [Event((Q(0),)) for _ in range(11)]
        response = reference_serial((lam,), impulse, (Q(0),), (Q(0),))
        for age, state in enumerate(response):
            assert state == ((1 - lam) * lam ** age,)
            count += 1
    return count


def check_scan_cases() -> tuple[int, int]:
    rng = random.Random(20261109)
    choices = (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(15, 16), Q(255, 256), Q(1))
    trials = 0
    states_checked = 0
    for length in (0, 1, 2, 3, 7, 8, 15, 31, 48):
        for width in (1, 3, 6):
            for _ in range(4):
                decays = tuple(rng.choice(choices) for _ in range(width))
                initial = tuple(Q(rng.randrange(-6, 7), 3) for _ in range(width))
                reset_state = tuple(Q(rng.randrange(-6, 7), 3) for _ in range(width))
                events = []
                for _ in range(length):
                    signal = tuple(Q(rng.randrange(-8, 9), 4) for _ in range(width))
                    valid = rng.random() >= 0.25
                    reset = rng.random() < 0.2
                    events.append(Event(signal, valid=valid, reset=valid and reset))
                pairs = [event_transform(decays, event, reset_state) for event in events]
                scanned = [apply(p, initial) for p in balanced_inclusive_scan(pairs, width)]
                direct = reference_serial(decays, events, initial, reset_state)
                assert scanned == direct
                for state in direct:
                    assert all(abs(v) <= 2 for v in state), "Convex bound failed"
                # Chunk execution must agree, including resets inside chunks.
                split = length // 2
                left = reference_serial(decays, events[:split], initial, reset_state)
                carry = left[-1] if left else initial
                right = reference_serial(decays, events[split:], carry, reset_state)
                assert left + right == direct
                trials += 1
                states_checked += length

    # Fixed candidate coefficients, represented exactly as their binary64 values.
    decays = tuple(Q.from_float(math.exp(-1.0 / tau))
                   for tau in (4, 32, 256) for _ in range(4))
    initial = (Q(1, 3),) * 12
    reset_state = (Q(0),) * 12
    events = [Event(tuple(Q(((i + k) % 7) - 3, 4) for k in range(12)),
                    valid=i not in (3, 9), reset=i in (0, 7, 12))
              for i in range(17)]
    pairs = [event_transform(decays, e, reset_state) for e in events]
    scanned = [apply(p, initial) for p in balanced_inclusive_scan(pairs, 12)]
    assert scanned == reference_serial(decays, events, initial, reset_state)
    trials += 1
    states_checked += len(events)

    # Padding must neither erase state nor advance the clock.
    lam = (Q(3, 4),)
    r0 = (Q(2, 3),)
    ignored = event_transform(lam, Event((Q(99),), valid=False), (Q(0),))
    assert ignored == identity(1)
    assert apply(ignored, r0) == r0
    # Reset+padding is outside the reference input domain and must be rejected.
    invalid_event = Event((Q(99),), valid=False, reset=True)
    for operation in (
        lambda: event_transform(lam, invalid_event, (Q(0),)),
        lambda: reference_serial(lam, [invalid_event], r0, (Q(0),)),
    ):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Reset on padding must be rejected")
    # A valid reset annihilates dependence on the pre-reset state.
    reset = event_transform(lam, Event((Q(1, 2),), reset=True), (Q(0),))
    assert reset.a == (Q(0),)
    assert apply(reset, (Q(-123),)) == apply(reset, (Q(456),)) == (Q(1, 8),)
    return trials, states_checked


def architecture_accounting() -> dict[str, int]:
    vocab, d, slots, operators, hidden, trace_width = 19, 32, 8, 4, 64, 12
    params = (2 * vocab * d + vocab + 5 * d * d + 4 * slots * d + 3 * slots
              + 2 * operators * d + operators + 5 * d * hidden + hidden + 10 * d + 7)
    macs = (5 * d * d + 8 * slots * d + 2 * operators * d + 5 * d * hidden
            + d * vocab + slots * slots + 7 * d)
    state_floats = slots * d + slots * slots + slots + 2 * d
    delta_params = 2 * d * trace_width + trace_width
    delta_macs = 2 * d * trace_width
    assert (params, macs, state_floats) == (18294, 18560, 392)
    assert (delta_params, delta_macs, 4 * trace_width) == (780, 768, 48)
    return {"base_parameters": params, "regulated_parameters": params + delta_params,
            "base_nominal_macs": macs, "regulated_nominal_macs": macs + delta_macs,
            "base_dense_state_bytes_fp32": 4 * state_floats,
            "regulated_dense_state_bytes_fp32": 4 * (state_floats + trace_width)}


def main() -> None:
    assoc = check_associativity()
    closed = check_closed_form()
    scan_trials, scanned_states = check_scan_cases()
    result = {"status": "passed", "arithmetic": "exact fractions",
              "associativity_triples": assoc, "closed_form_and_impulse_checks": closed,
              "scan_trials": scan_trials, "scanned_state_vectors": scanned_states,
              "checks": ["identity", "chronological_noncommutativity", "associativity",
                         "closed_form", "impulse_response", "serial_scan_equality",
                         "padding_identity", "reset_forgets_prior_state", "chunk_equivalence",
                         "reset_padding_rejected", "bounded_states", "candidate_coefficient_approximations"],
              "accounting": architecture_accounting(),
              "limitations": ["Finite executable checks are not formal verification.",
                              "No FP32 numerical parity or throughput is measured.",
                              "No language, retrieval, or controller benefit is established."]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
