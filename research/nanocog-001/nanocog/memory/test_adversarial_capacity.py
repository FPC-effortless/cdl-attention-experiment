"""SQLite capacity failures must not acknowledge partial memory operations."""
import sqlite3

import pytest

from .store import MemoryStore


def cap_at_current_pages(store):
    # A real SQLite allocation limit raises SQLITE_FULL in the engine. It is a
    # deterministic capacity injection, not an emulated filesystem power loss.
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    pages = store.db.execute("PRAGMA page_count").fetchone()[0]
    applied = store.db.execute(f"PRAGMA max_page_count={pages}").fetchone()[0]
    assert applied == pages


def release_cap(store):
    store.db.execute("PRAGMA max_page_count=1073741823")


def large_source():
    return " ".join(f"syntheticcapacitytoken{i:05d}" for i in range(10000))


def test_disk_full_message_rolls_back_and_exact_request_can_retry(tmp_path):
    path = tmp_path / "messages.sqlite3"
    with MemoryStore(path, principal_id="capacity") as store:
        store.append_message("s", "user", "Original accepted observation", event_id="original")
        cap_at_current_pages(store)
        with pytest.raises(sqlite3.OperationalError, match="full"):
            store.append_message("s", "user", large_source(), event_id="large")
        assert not store.db.in_transaction
        assert store.get_message("large") is None
        assert [m["event_id"] for m in store.messages()] == ["original"]
        assert store.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        release_cap(store)
        saved = store.append_message("s", "user", large_source(), event_id="large")
        assert saved["event_id"] == "large"
    with MemoryStore(path, principal_id="capacity") as reopened:
        assert reopened.get_message("large")["content"] == large_source()
        assert len(reopened.messages()) == 2


def test_disk_full_fact_acceptance_preserves_old_revision_and_pending_proposal(tmp_path):
    path = tmp_path / "facts.sqlite3"
    with MemoryStore(path, principal_id="capacity") as store:
        store.append_message("s", "user", "Inventory observation", event_id="source")
        store.propose("inventory", {"value": "old"}, "source", proposal_id="initial")
        store.accept("initial", authority="test-host")
        store.propose("inventory", {"value": large_source()}, "source", proposal_id="replacement", expected_revision=1)
        cap_at_current_pages(store)
        with pytest.raises(sqlite3.OperationalError, match="full"):
            store.accept("replacement", authority="test-host")
        assert not store.db.in_transaction
        assert store.fact_revision("inventory") == 1
        assert store.facts("inventory")[0]["value"] == {"value": "old"}
        assert len(store.history("inventory")) == 1
        pending = store.db.execute("SELECT status FROM proposals WHERE proposal='replacement'").fetchone()[0]
        assert pending == "proposed"
        release_cap(store)
        result = store.accept("replacement", authority="test-host")
        assert result["revision"] == 2
    with MemoryStore(path, principal_id="capacity") as reopened:
        assert reopened.facts("inventory")[0]["value"] == {"value": large_source()}
        assert len(reopened.history("inventory")) == 2
