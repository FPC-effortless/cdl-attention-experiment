"""Stage-0 PNDS-GATE-001: enforce the no-answer/no-gold-structure interface.

This stage deliberately does not claim end-to-end PNDS evidence. It verifies that
router inputs can be represented by an explicit allowlist and that causal
interventions are observable before training a persistent decision loop.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

FORBIDDEN_FIELDS = frozenset({
    "target", "answer", "outcome", "reward", "true_edge_set", "true_edges",
    "oracle_mask", "gold_structure", "structure_id", "correct_action",
})
ALLOWED_QUERY_FIELDS = frozenset({"query", "query_embedding", "observation"})
ALLOWED_STATE_FIELDS = frozenset({"candidate_keys", "candidate_values", "candidate_provenance", "validity"})

@dataclass(frozen=True)
class InterfaceAudit:
    passed: bool
    forbidden_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    allowed_fields: tuple[str, ...]


def audit_router_schema(payload: dict[str, Any]) -> InterfaceAudit:
    keys = set(payload)
    forbidden = tuple(sorted(keys & FORBIDDEN_FIELDS))
    allowed = ALLOWED_QUERY_FIELDS | ALLOWED_STATE_FIELDS
    unknown = tuple(sorted(keys - allowed - FORBIDDEN_FIELDS))
    return InterfaceAudit(
        passed=not forbidden and not unknown,
        forbidden_fields=forbidden,
        unknown_fields=unknown,
        allowed_fields=tuple(sorted(keys & allowed)),
    )


def assert_no_gold_router_input(payload: dict[str, Any]) -> None:
    audit = audit_router_schema(payload)
    if not audit.passed:
        raise AssertionError(
            "PNDS-GATE-001 router interface violation: "
            f"forbidden={audit.forbidden_fields}, unknown={audit.unknown_fields}"
        )


def causal_intervention_delta(success_before: float, success_after: float) -> float:
    """Observed intervention effect: P(success|selected)-P(success|intervened)."""
    return float(success_before) - float(success_after)


def verify_state_update(*, verified: bool, proposed_update: dict[str, Any]) -> dict[str, Any]:
    """Reference commit rule: unverified outcomes cannot mutate persistent state."""
    if not verified:
        return {"committed": False, "reason": "verification_failed", "update": None}
    return {"committed": True, "reason": "verified", "update": proposed_update}


if __name__ == "__main__":
    # Positive control: valid interface.
    assert_no_gold_router_input({
        "query": "q",
        "observation": "o",
        "candidate_keys": ["k0", "k1"],
        "candidate_values": ["v0", "v1"],
        "candidate_provenance": ["p0", "p1"],
        "validity": [1, 1],
    })
    # Negative controls: each must be rejected.
    for bad in ("target", "outcome", "true_edge_set", "gold_structure", "correct_action"):
        try:
            assert_no_gold_router_input({"query": "q", bad: 1})
        except AssertionError:
            continue
        raise AssertionError(f"forbidden field was accepted: {bad}")
    assert verify_state_update(verified=False, proposed_update={"x": 1})["committed"] is False
    assert verify_state_update(verified=True, proposed_update={"x": 1})["committed"] is True
    print("PNDS-GATE-001 preflight: PASS")
