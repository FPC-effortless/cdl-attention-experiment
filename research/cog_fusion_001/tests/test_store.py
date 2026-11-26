"""Adversarial executor tests; these do not qualify a learned language model."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.store import (CapacityError, ConflictError, InjectedFailure, IntegrityError,
                          Operation, Store, digest)


OWNER = ("w", "chest", "owner")
CONTAINER = ("w", "coin", "container")
OTHER = ("w", "crate", "owner")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "facts.sqlite", "trained-model-digest", capacity=64)
    yield s
    s.close()


def write(s, request, key, value, *, source=None, kind="set", **kwargs):
    return s.commit_request(request, {"key": key, "value": value, "kind": kind},
                            operations=[Operation(kind, key, value, source or request)], **kwargs)


def initial(s):
    write(s, "owner", OWNER, "Amara", neural_state=b"s1", rng_state=b"r1")
    write(s, "link", CONTAINER, "chest", kind="link")
    write(s, "other", OTHER, "Cato")


def test_actual_two_hop_and_unrelated_preservation(store):
    initial(store)
    packet = store.resolve(CONTAINER, "owner")
    assert packet.value == "Amara"
    assert packet.values == ("chest", "Amara")
    assert packet.read_count == 2
    assert tuple(d.key for d in packet.dependencies) == (CONTAINER, OWNER)
    write(store, "unrelated", OTHER, "Dara")
    assert store.validate_packet(packet)
    write(store, "reroute", CONTAINER, "crate", kind="link")
    assert not store.validate_packet(packet)
    new = store.resolve(CONTAINER, "owner")
    assert new.value == "Dara"
    assert tuple(d.key for d in new.dependencies) == (CONTAINER, OTHER)
    assert store.resolve(OWNER).value == "Amara"


def test_capacity_is_current_identity_not_number_of_events(tmp_path):
    s = Store(tmp_path / "capacity.db", "m", capacity=2)
    write(s, "a", OWNER, "Amara")
    write(s, "b", CONTAINER, "chest", kind="link")
    for i in range(30):
        write(s, f"overwrite-{i}", OWNER, "Amara" if i % 2 else "Bela")
    old = s.snapshot_state()
    with pytest.raises(CapacityError):
        s.commit_request("overflow", "overflow", operations=[
            Operation("set", OWNER, "Wrong", "source-1"),
            Operation("set", OTHER, "Unexpected", "source-2")],
            neural_state=b"must-not-publish", rng_state=b"must-not-publish")
    assert s.snapshot_state() == old
    assert s.resolve(OWNER).value == "Amara"
    assert s.resolve(OTHER).status == "missing"
    s.commit_request("delete", "delete", operations=[Operation("delete", OWNER)])
    write(s, "freed", OTHER, "Cato")
    assert len(s.current_records()) == 2
    s.close()


def test_missing_delete_recreate_and_aba_invalidate(store):
    missing = store.resolve(OWNER)
    assert missing.status == "missing" and store.validate_packet(missing)
    write(store, "a", OWNER, "Amara")
    assert not store.validate_packet(missing)
    first = store.resolve(OWNER)
    write(store, "b", OWNER, "Bela")
    write(store, "a-again", OWNER, "Amara")
    assert not store.validate_packet(first)
    restored = store.resolve(OWNER)
    store.commit_request("del", "delete", operations=[Operation("delete", OWNER)])
    deleted = store.resolve(OWNER)
    assert deleted.status == "deleted"
    assert not store.validate_packet(restored)
    write(store, "new-incarnation", OWNER, "Amara")
    assert not store.validate_packet(deleted)
    assert not store.validate_packet(first)
    assert store.resolve(OWNER).value == "Amara"


def test_source_erasure_and_copy_provenance(store):
    write(store, "a", OWNER, "Amara", source="observation-A")
    store.commit_request("copy", "copy", operations=[
        Operation("copy", OTHER, source="copy-observation", copy_from=OWNER)])
    original, copied = store.resolve(OWNER), store.resolve(OTHER)
    write(store, "change-original", OWNER, "Bela", source="observation-B")
    assert store.resolve(OTHER).value == "Amara"  # COPY is a value snapshot.
    store.invalidate_source("observation-A", request_id="erase")
    assert not store.validate_packet(original)
    assert not store.validate_packet(copied)
    assert store.resolve(OTHER).status == "retracted"
    assert store.resolve(OWNER).value == "Bela"
    with pytest.raises(ConflictError):
        write(store, "resurrect-erased-source", OTHER, "Amara", source="observation-A")
    write(store, "fresh-source", OTHER, "Amara", source="new-observation")
    assert store.resolve(OTHER).value == "Amara"


def test_pending_blocks_relevant_dependencies_and_unknown_scope(store):
    initial(store)
    previous = store.resolve(CONTAINER, "owner")
    store.mark_pending("correction", key=OWNER, payload="Bela now owns chest")
    blocked = store.resolve(CONTAINER, "owner")
    assert blocked.status == "pending" and blocked.value is None
    assert blocked.values == ("chest",)
    assert blocked.pending == ("correction",)
    assert store.resolve(OTHER).status == "ok"
    assert not store.validate_packet(previous)
    with pytest.raises(InjectedFailure):
        store.commit_request("retry", "correction", operations=[Operation("set", OWNER, "Bela")],
                             clear_pending=["correction"], failpoint="before_commit")
    assert store.resolve(CONTAINER, "owner").status == "pending"
    store.commit_request("retry", "correction", operations=[Operation("set", OWNER, "Bela")],
                         clear_pending=["correction"])
    assert store.resolve(CONTAINER, "owner").value == "Bela"
    assert not store.validate_packet(blocked)
    store.mark_pending("uninterpreted", payload="affected scope is unknown")
    assert store.resolve(OTHER).status == "pending"
    assert store.resolve(("another-scope", "missing", "owner")).status == "pending"
    store.clear_pending("uninterpreted", request_id="resolved-global")
    assert store.resolve(OTHER).value == "Cato"


def test_pending_aba_and_idempotent_intent(store):
    write(store, "a", OWNER, "Amara")
    packet = store.resolve(OWNER)
    first = store.mark_pending("p", "w", payload="intent")
    second = store.mark_pending("p", "w", payload="intent")
    assert second["replayed"] and first["revision"] == second["revision"]
    with pytest.raises(ConflictError):
        store.mark_pending("p", "other", payload="intent")
    store.clear_pending("p", request_id="clear")
    assert store.resolve(OWNER).value == "Amara"
    assert not store.validate_packet(packet)


def test_completed_pending_intent_cannot_be_resurrected_by_full_retry(store):
    write(store, "a", OWNER, "Amara")
    store.mark_pending("change", key=OWNER, payload="Bela")
    store.commit_request("commit-change", "Bela", operations=[Operation("set", OWNER, "Bela")],
                         clear_pending=["change"])
    before = store.snapshot_state()
    repeated_mark = store.mark_pending("change", key=OWNER, payload="Bela")
    assert repeated_mark["replayed"] and repeated_mark["resolved"]
    repeated_commit = store.commit_request("commit-change", "Bela", clear_pending=["change"])
    assert repeated_commit["replayed"]
    assert store.resolve(OWNER).status == "ok" and store.resolve(OWNER).value == "Bela"
    assert store.snapshot_state() == before
    with pytest.raises(ConflictError):
        store.mark_pending("change", key=OWNER, payload="Cato")


def test_lookup_receipt_is_early_read_only_idempotency_check(store):
    assert store.lookup_receipt("absent", "input") is None
    original = store.commit_request("done", "input", reply="response", neural_state=b"saved")
    before = store.snapshot_state()
    assert store.lookup_receipt("done", "input") == dict(original, replayed=True)
    assert store.snapshot_state() == before
    with pytest.raises(ConflictError):
        store.lookup_receipt("done", "other")


def test_same_revision_forks_have_distinct_dependency_identity(store, tmp_path):
    write(store, "a", OWNER, "Amara")
    left = store.fork(tmp_path / "left.db")
    right = store.fork(tmp_path / "right.db")
    try:
        write(left, "fork-edit", OWNER, "Bela")
        write(right, "fork-edit", OWNER, "Cato")
        p, q = left.resolve(OWNER), right.resolve(OWNER)
        assert p.revision == q.revision
        assert p.dependencies[0].version == q.dependencies[0].version
        assert p.value != q.value
        assert not left.validate_packet(q) and not right.validate_packet(p)
        assert not left.validate_packet(store.resolve(OWNER))
    finally:
        left.close(); right.close()


def test_cas_rejects_stale_neural_continuation_and_snapshot(store):
    peer = Store(store.path, store.model_id)
    old = store.snapshot_state()
    write(peer, "external", OWNER, "Bela", neural_state=b"new", rng_state=b"new-rng")
    with pytest.raises(ConflictError):
        store.snapshot_state()
    with pytest.raises(ConflictError):
        store.commit_request("stale", "stale", neural_state=b"old", rng_state=b"old-rng")
    with pytest.raises(ConflictError):
        store.snapshot_state(expected_revision=old["revision"])
    fresh = store.refresh_snapshot()
    assert fresh["neural_state"] == b"new" and fresh["rng_state"] == b"new-rng"
    store.commit_request("new", "new", neural_state=b"continuation", expected_revision=fresh["revision"])
    assert store.snapshot_state()["neural_state"] == b"continuation"
    peer.close()


@pytest.mark.parametrize("stage", ["before_operations", "after_operations", "before_commit"])
def test_injected_failures_rollback_facts_state_rng_reply_and_pending(store, stage):
    initial(store)
    store.mark_pending("p", key=OWNER)
    before = store.snapshot_state()
    with pytest.raises(InjectedFailure):
        store.commit_request("candidate", "payload", operations=[Operation("set", OWNER, "Bela")],
                             clear_pending=["p"], neural_state=b"new-neural", rng_state=b"new-rng",
                             reply="Bela", failpoint=stage)
    assert store.snapshot_state() == before
    assert store.resolve(OWNER).status == "pending"
    assert store.db.execute("SELECT COUNT(*) FROM receipts WHERE request_id='candidate'").fetchone()[0] == 0
    store.commit_request("candidate", "payload", operations=[Operation("set", OWNER, "Bela")],
                         clear_pending=["p"], neural_state=b"new-neural", rng_state=b"new-rng", reply="Bela")
    assert store.resolve(OWNER).value == "Bela"


def test_lost_reply_retries_exact_historical_receipt_without_mutating_new_state(store):
    with pytest.raises(InjectedFailure):
        write(store, "request", OWNER, "Amara", neural_state=b"s1", rng_state=b"r1", reply="Amara",
              failpoint="after_commit")
    write(store, "newer", OWNER, "Bela", neural_state=b"s2", rng_state=b"r2")
    before = store.snapshot_state()
    replay = write(store, "request", OWNER, "Amara", neural_state=b"bad", rng_state=b"bad", reply="bad")
    assert replay["replayed"] and replay["historical_receipt"]
    assert replay["reply"] == "Amara" and replay["revision"] == 1
    assert store.snapshot_state() == before
    assert store.resolve(OWNER).value == "Bela"
    with pytest.raises(ConflictError):
        store.commit_request("request", "different input")


@pytest.mark.parametrize("stage,expected", [("after_operations", "Amara"), ("after_commit", "Bela")])
def test_real_process_crash_recovery(tmp_path, stage, expected):
    path = tmp_path / (stage + ".db")
    with Store(path, "m") as s:
        write(s, "before", OWNER, "Amara", neural_state=b"old", rng_state=b"oldrng")
    script = """
import os, sys
from fusion.store import Store, Operation
s=Store(sys.argv[1], 'm')
def crash(stage):
    if stage==sys.argv[2]: os._exit(73)
s.commit_request('crashed', 'payload', operations=[Operation('set',('w','chest','owner'),'Bela')],
                 neural_state=b'new',rng_state=b'newrng',reply='Bela',failpoint=crash)
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.run([sys.executable, "-c", script, str(path), stage], env=environment, check=False)
    assert proc.returncode == 73
    with Store(path, "m") as s:
        assert s.resolve(OWNER).value == expected
        snapshot = s.snapshot_state()
        assert snapshot["neural_state"] == (b"old" if expected == "Amara" else b"new")
        assert snapshot["rng_state"] == (b"oldrng" if expected == "Amara" else b"newrng")
        receipt = s.commit_request("crashed", "payload", operations=[Operation("set", OWNER, "Bela")],
                                   neural_state=b"new", rng_state=b"newrng", reply="Bela")
        assert receipt["replayed"] == (expected == "Bela")
        assert s.resolve(OWNER).value == "Bela"
        assert s.db.execute("SELECT COUNT(*) FROM events WHERE request_id='crashed'").fetchone()[0] == 1


def test_packet_reply_dependency_cas_and_forged_packet(store):
    write(store, "a", OWNER, "Amara")
    packet = store.resolve(OWNER)
    forged = replace(packet, value="Bela", values=("Bela",), packet_digest="")
    forged = replace(forged, packet_digest=digest(forged.to_dict()))
    assert not store.validate_packet(forged)
    with pytest.raises(ConflictError):
        store.commit_request("forged", "query", reply="Bela", packet=forged)
    assert store.commit_request("answer", "query", reply="Amara", packet=packet)["reply"] == "Amara"
    write(store, "correction", OWNER, "Bela")
    with pytest.raises(ConflictError):
        store.commit_request("stale-answer", "new query", reply="Amara", packet=packet)
    # A fact mutation and its old reply cannot sneak into one transaction.
    packet = store.resolve(OWNER)
    with pytest.raises(ConflictError):
        store.commit_request("mixed", "mixed", operations=[Operation("set", OWNER, "Cato")],
                             reply="Bela", packet=packet)
    assert store.resolve(OWNER).value == "Bela"


def test_bounded_candidate_overflow_tie_missing_and_pending(store):
    initial(store)
    keys = [OWNER, OTHER, ("w", "missing", "owner"), CONTAINER]
    selected = store.select_candidates(keys, [0.9, 0.2, 1.0, 0.1])
    assert selected.status == "ok" and selected.selected_value == "Amara"
    assert selected.candidates_inspected == 4 and len(selected.dependencies) == 4
    before = dict(store.counters)
    overflow = store.select_candidates(keys + [("w", "five", "owner")], [1, 2, 3, 4, 5])
    assert overflow.status == "overflow" and overflow.candidates_inspected == 0
    assert store.counters == before
    assert store.select_candidates([OWNER, OTHER], [0.5, 0.5]).status == "tie"
    assert store.select_candidates([keys[2]], [1]).status == "missing"
    store.mark_pending("p", key=OTHER)
    assert store.select_candidates([OWNER, OTHER], [0.9, 0.1]).status == "pending"
    with pytest.raises(ValueError):
        store.select_candidates([OWNER, OWNER], [1, 0])
    with pytest.raises(ValueError):
        store.select_candidates([OWNER], [float("nan")])


def test_scopes_and_invalid_links(store):
    write(store, "one", OWNER, "Amara")
    write(store, "two", ("other", "chest", "owner"), "Bela")
    assert store.resolve(OWNER).value == "Amara"
    assert store.resolve(("other", "chest", "owner")).value == "Bela"
    write(store, "not-a-reference", CONTAINER, 123)
    p = store.resolve(CONTAINER, "owner")
    assert p.status == "invalid_link" and p.read_count == 1


def test_interpretation_and_corrupt_state_fail_closed(store):
    with pytest.raises(IntegrityError):
        Store(store.path, "different-model")
    with pytest.raises(IntegrityError):
        Store(store.path, store.model_id, schema_id="different-schema")
    write(store, "a", OWNER, "Amara", neural_state=b"sound")
    store.db.execute("UPDATE runtime SET neural_state=?", (b"corrupt",))
    with pytest.raises(IntegrityError):
        store.snapshot_state()
    with pytest.raises(IntegrityError):
        store.commit_request("overwrite-corrupt", "request", neural_state=b"new")


def test_corrupt_record_and_metadata_fail_closed(store):
    write(store, "a", OWNER, "Amara")
    store.db.execute("UPDATE records SET value=?", (json.dumps("Bela"),))
    with pytest.raises(IntegrityError):
        store.resolve(OWNER)
    with pytest.raises(IntegrityError):
        write(store, "overwrite-corrupt", OWNER, "Cato")
    store.db.execute("UPDATE metadata SET value='another-branch' WHERE name='branch_id'")
    with pytest.raises(IntegrityError):
        store.snapshot_state()


def test_two_hop_access_budget_independent_of_other_records(tmp_path):
    with Store(tmp_path / "scale.db", "m", capacity=1024) as s:
        initial(s)
        operations = [Operation("set", ("w", f"distractor-{i}", "owner"), "Dara", f"source-{i}")
                      for i in range(1000)]
        s.commit_request("many", "many", operations=operations)
        before = s.counters["record_reads"]
        p = s.resolve(CONTAINER, "owner")
        assert p.value == "Amara"
        assert p.read_count == 2 and s.counters["record_reads"] - before == 2
        before = s.counters["validation_reads"]
        assert s.validate_packet(p)
        assert s.counters["validation_reads"] - before == 2


def test_nonfinite_and_partial_batch_rejection_is_atomic(store):
    before = store.snapshot_state()
    with pytest.raises(ValueError):
        store.commit_request("bad", "bad", operations=[
            Operation("set", OWNER, "Amara"), Operation("set", OTHER, float("nan"))])
    assert store.snapshot_state() == before
    assert store.current_records() == {}


def test_restart_returns_exact_opaque_continuation_and_receipt(tmp_path):
    path = tmp_path / "restart.db"
    state = bytes(range(256)) * 8
    rng = bytes(reversed(range(256)))
    with Store(path, "m") as s:
        write(s, "observe", OWNER, "Amara", neural_state=state, rng_state=rng)
        packet = s.resolve(OWNER)
        receipt = s.commit_request("answer", "query", reply={"text": "Amara"}, packet=packet)
        snapshot = s.snapshot_state()
    with Store(path, "m") as resumed:
        assert resumed.snapshot_state() == snapshot
        assert resumed.snapshot_state()["neural_state"] == state
        assert resumed.snapshot_state()["rng_state"] == rng
        replay = resumed.commit_request("answer", "query")
        assert replay == dict(receipt, replayed=True)
        assert resumed.validate_packet(packet)
