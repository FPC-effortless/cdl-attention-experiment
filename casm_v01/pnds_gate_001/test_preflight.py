from .preflight import audit_router_schema, causal_intervention_delta, verify_state_update


def test_valid_router_schema():
    result = audit_router_schema({
        "query": "q", "observation": "o", "candidate_keys": [],
        "candidate_values": [], "candidate_provenance": [], "validity": [],
    })
    assert result.passed


def test_target_and_gold_structure_are_rejected():
    for key in ("target", "answer", "outcome", "true_edge_set", "gold_structure", "correct_action"):
        result = audit_router_schema({"query": "q", key: 1})
        assert not result.passed
        assert key in result.forbidden_fields


def test_unknown_interface_fields_are_rejected():
    result = audit_router_schema({"query": "q", "hidden_shortcut": 1})
    assert not result.passed
    assert "hidden_shortcut" in result.unknown_fields


def test_intervention_delta():
    assert causal_intervention_delta(0.9, 0.2) == 0.7


def test_unverified_update_is_not_committed():
    assert not verify_state_update(verified=False, proposed_update={"x": 1})["committed"]


def test_verified_update_is_committed():
    result = verify_state_update(verified=True, proposed_update={"x": 1})
    assert result["committed"]
    assert result["update"] == {"x": 1}
