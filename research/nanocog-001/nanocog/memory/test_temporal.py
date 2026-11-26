"""Typed-reference checks; these tests make no learned generalization claim."""
from dataclasses import replace
import json
import random

import pytest

from nanocog.memory.temporal import (
    AFTER, BEFORE, UNKNOWN, SourceSpan, TemporalClaim, TemporalReference,
    check_consistency,
)


def claim(a="leave", b="arrive", *, relation=BEFORE, status="accepted", source="m1"):
    cue = "before" if relation == BEFORE else "after"
    text = f"Éva left {cue} Tom arrived."
    anchor = text.index("arrived")
    item = TemporalClaim(a, relation, b, source,
                         SourceSpan(9, 9 + len(cue), cue), SourceSpan(4, 8, "left"),
                         SourceSpan(anchor, anchor + 7, "arrived"), status)
    return item, {source: text}


def test_default_is_proposal_and_cannot_establish_order():
    item, sources = claim()
    proposed = TemporalClaim(item.action_id, item.relation, item.anchor_action_id,
                             item.source_event_id, *item.spans)
    assert proposed.status == "proposed"
    ref = TemporalReference([proposed], sources)
    assert ref.relation_between("leave", "arrive").relation == UNKNOWN
    assert ref.consistency.accepted_claims == 0
    assert ref.consistency.ignored_claims == 1


def test_exact_unicode_spans_and_strict_json_roundtrip():
    item, sources = claim()
    assert item.validate(sources["m1"]) is item
    data = json.loads(json.dumps(item.to_dict()))
    assert TemporalClaim.from_dict(data) == item
    wrong_source = sources["m1"].replace("before", "after ")
    with pytest.raises(ValueError, match="span"):
        TemporalReference([item], {"m1": wrong_source})
    with pytest.raises(ValueError, match="missing source"):
        TemporalReference([item], {})
    with pytest.raises(ValueError, match="recording time"):
        TemporalClaim.from_dict({**data, "time_basis": "recording"})
    for bad in (True, 2, "1"):
        with pytest.raises(ValueError, match="schema"):
            TemporalClaim.from_dict({**data, "schema_version": bad})
    with pytest.raises(ValueError, match="serialized"):
        TemporalClaim.from_dict({**data, "event_phase": 0})


@pytest.mark.parametrize("args", [(True, 2, "a"), (-1, 1, "ab"), (2, 2, ""),
                                 (1, 3, "a"), (0, 1, 9)])
def test_malformed_spans_are_rejected(args):
    with pytest.raises(ValueError, match="span"):
        SourceSpan(*args)


@pytest.mark.parametrize("change", [{"relation": "EARLY"}, {"status": "trusted"},
                                   {"action_id": "arrive"}, {"action_id": ""},
                                   {"source_event_id": " m1"}, {"anchor_span": (20, 27)}])
def test_unbound_or_malformed_claims_are_rejected(change):
    item, _ = claim()
    with pytest.raises(ValueError):
        replace(item, **change)


def test_before_after_inverse_and_transitive_evidence():
    first, sources = claim("A", "B", source="late_record")
    second, other = claim("C", "B", relation=AFTER, source="early_record")
    sources.update(other)
    ref = TemporalReference([second, first], sources)
    assert ref.consistency.consistent
    assert ref.relation_between("A", "C").relation == BEFORE
    assert ref.relation_between("C", "A").relation == AFTER
    assert ref.relation_between("A", "C").source_event_ids == ("early_record", "late_record")
    assert ref.relation_between("A", "A").relation == UNKNOWN
    assert ref.relation_between("missing", "A").relation == UNKNOWN
    # Permuting recording order does not change the described world's order.
    reverse = TemporalReference([first, second], sources)
    assert reverse.relation_between("A", "C") == ref.relation_between("A", "C")


def test_uniqueness_requires_bound_exclusions_and_never_breaks_ties_by_order():
    before, sources = claim("A", "anchor", source="one")
    after, other = claim("B", "anchor", relation=AFTER, source="two")
    sources.update(other)
    ref = TemporalReference([before, after], sources)
    selected = ref.select_event(["B", "A"], BEFORE, "anchor")
    assert selected.action_id == "A" and selected.status == "known"
    assert selected.source_event_ids == ("one", "two")
    assert ref.select_event(["A", "B"], AFTER, "anchor").action_id == "B"
    assert ref.select_event(["A", "unknown"], BEFORE, "anchor").action_id is None
    assert ref.select_event([], BEFORE, "anchor").reason == "no_candidates"
    assert ref.select_event(["A"], AFTER, "anchor").reason == "no_match"
    with pytest.raises(ValueError, match="distinct"):
        ref.select_event(["A", "A"], BEFORE, "anchor")
    with pytest.raises(ValueError, match="iterable"):
        ref.select_event("A", BEFORE, "anchor")
    also, other = claim("C", "anchor", source="three")
    sources.update(other)
    ambiguous = TemporalReference([before, after, also], sources)
    for order in (["A", "C"], ["C", "A"]):
        assert ambiguous.select_event(order, BEFORE, "anchor").reason == "ambiguous"


def test_unordered_accepted_components_and_explicit_ambiguity_stay_unknown():
    first, sources = claim("A", "B", source="one")
    second, other = claim("C", "D", source="two")
    sources.update(other)
    ref = TemporalReference([first, second], sources)
    assert ref.relation_between("A", "D").reason == "unordered"
    for status in ("proposed", "ambiguous", "rejected"):
        candidate = TemporalReference([replace(first, status=status)], sources)
        assert candidate.select_event(["A"], BEFORE, "B").status == "unknown"


def test_cycles_including_transitive_contradictions_fail_closed():
    pairs = [("A", "B"), ("B", "C"), ("C", "A")]
    claims, sources = [], {}
    for i, (a, b) in enumerate(pairs):
        item, text = claim(a, b, source=f"s{i}")
        claims.append(item)
        sources.update(text)
    expected = check_consistency(claims, sources)
    assert not expected.consistent
    assert expected.cycle == ("A", "B", "C", "A")
    for ordered in (claims, list(reversed(claims))):
        ref = TemporalReference(ordered, sources)
        assert ref.consistency == expected
        assert ref.relation_between("A", "B").reason == "inconsistent"
        assert ref.select_event(["A"], BEFORE, "B").reason == "inconsistent"
    assert check_consistency(claims[:2], sources).consistent
    contradiction = replace(claims[0], relation=AFTER)
    assert not check_consistency([claims[0], contradiction], sources).consistent


def test_proposed_conflict_cannot_overwrite_an_accepted_binding():
    item, sources = claim()
    conflict = replace(item, relation=AFTER, status="proposed")
    ref = TemporalReference([item, conflict], sources)
    assert ref.consistency.consistent
    assert ref.relation_between("leave", "arrive").relation == BEFORE


def test_bound_relation_swap_changes_choice_and_unrelated_relation_does_not():
    first, sources = claim("A", "anchor", source="first")
    second, other = claim("B", "anchor", relation=AFTER, source="second")
    sources.update(other)
    ref = TemporalReference([first, second], sources)
    assert ref.select_event(["A", "B"], BEFORE, "anchor").action_id == "A"
    assert ref.select_event(["A", "B"], AFTER, "anchor").action_id == "B"
    swapped_a, swapped_sources = claim("A", "anchor", relation=AFTER, source="first")
    swapped_b, other = claim("B", "anchor", relation=BEFORE, source="second")
    swapped_sources.update(other)
    swapped = TemporalReference([swapped_a, swapped_b], swapped_sources)
    assert swapped.select_event(["A", "B"], BEFORE, "anchor").action_id == "B"
    assert swapped.select_event(["A", "B"], AFTER, "anchor").action_id == "A"
    irrelevant, other = claim("X", "Y", source="unrelated")
    sources.update(other)
    augmented = TemporalReference([first, irrelevant, second], sources)
    assert augmented.select_event(["A", "B"], BEFORE, "anchor") == ref.select_event(
        ["A", "B"], BEFORE, "anchor")


def test_long_history_is_iterative_and_replay_deterministic():
    claims, sources = [], {}
    for i in range(1200):
        item, text = claim(f"action-{i:04d}", f"action-{i+1:04d}", source=f"message-{i:04d}")
        claims.append(item)
        sources.update(text)
    random.Random(15).shuffle(claims)
    ref = TemporalReference(claims, sources)
    result = ref.relation_between("action-0000", "action-1200")
    assert result.relation == BEFORE and len(result.source_event_ids) == 1200
    assert ref.consistency.consistent
    assert ref.relation_between("action-1200", "action-0000").relation == AFTER
