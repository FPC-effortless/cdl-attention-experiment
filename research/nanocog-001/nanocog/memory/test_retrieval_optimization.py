"""Exact lexical semantics must survive the retrieval hot-path optimization."""
from __future__ import annotations

import random
import re
import unicodedata
from unittest.mock import patch

import pytest

from . import store as implementation
from .store import ConflictError, MemoryStore


def baseline_tokens(text):
    """Literal pre-optimization algorithm, retained as a semantic oracle."""
    folded = ''.join(c for c in unicodedata.normalize('NFD', text.casefold())
                     if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", folded, flags=re.UNICODE)


def baseline_score(content, query):
    words = baseline_tokens(content)
    overlap = len(set(words).intersection(frozenset(baseline_tokens(query))))
    return overlap + overlap / (1 + len(words))


SAMPLES = (
    "", "A_a B-2 0000 repeated repeated", "Orion launches Thursday; stock 17.",
    "Straße STRASSE \u017f \u212a", "café CAFÉ cafe\u0301", "中文 日本語 한국어",
    "Ελληνικά Καλημέρα русский ё", "عربي فارسی اردو", "ọjọ́ Yorùbá Ìbàdàn",
    "\u0345\u0301 A\u0327\u0308", "emoji🙂 punctuation—‘quoted’", "__ ___ 0\x00x",
    "① ² Ⅷ １２３", "a " * 1000,
)


@pytest.mark.parametrize("text", SAMPLES)
def test_tokens_and_scores_match_preoptimization_unicode_algorithm(text):
    assert implementation._word_tokens(text) == baseline_tokens(text)
    for query in SAMPLES:
        assert implementation._lexical_score(text, query) == baseline_score(text, query)


def test_random_unicode_and_ascii_have_exact_tokens_and_scores():
    rng = random.Random(41173)
    alphabet = list(" abcdefXYZ_0123!?-\t\n") + list("éßſKọ中Ω🙂") + ["\u0301", "\u0327", "\u0345"]
    for _ in range(1000):
        text = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(160)))
        query = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(48)))
        assert implementation._word_tokens(text) == baseline_tokens(text)
        assert implementation._lexical_score(text, query) == baseline_score(text, query)
    for _ in range(1000):
        text = ''.join(chr(rng.randrange(0x110000)) for _ in range(rng.randrange(32)))
        assert implementation._word_tokens(text) == baseline_tokens(text)


def test_ascii_and_casefold_to_ascii_do_not_require_unicode_character_scan():
    with patch.object(implementation.unicodedata, "normalize", side_effect=AssertionError("ASCII needs no decomposition")), \
         patch.object(implementation.unicodedata, "combining", side_effect=AssertionError("ASCII has no combining marks")):
        assert implementation._word_tokens("Orion THURSDAY_3") == ["orion", "thursday", "3"]
        assert implementation._word_tokens("Straße \u017f \u212a") == ["strasse", "s", "k"]


def compare_with_old_ranker(store, call):
    store.db.create_function("cog_lexical_score", 2, baseline_score, deterministic=True)
    try:
        expected = call()
    finally:
        store.db.create_function("cog_lexical_score", 2, implementation._lexical_score, deterministic=True)
    actual = call()
    assert actual == expected


def exercise_ranked_calls(store):
    queries = ["Orion launch", "Thursday", "cafe", "café", "Straße", "STRASSE", "中文",
               "Ελληνικά", "Yorùbá", "🙂", "__", "unknown", "", "stock repeated",
               " ".join(f"token{i}" for i in range(40))]
    for query in queries:
        for limit in (1, 8, 31):
            compare_with_old_ranker(store, lambda: store.messages(query=query, limit=limit))
            compare_with_old_ranker(store, lambda: store.messages("session1", query=query, limit=limit,
                                                               exclude_event_id="event17"))
            compare_with_old_ranker(store, lambda: store.recall(query, limit=min(limit, 20), max_chars=8000))
        compare_with_old_ranker(store, lambda: store.recall(query, limit=8, max_chars=256,
                                                           exclude_event_id="event17"))


def test_database_ranking_context_and_provenance_match_before_after_mutations(tmp_path):
    path = tmp_path / "ranking.sqlite3"
    with MemoryStore(path, principal_id="alice", namespace="work") as store:
        for i in range(120):
            content = SAMPLES[1 + i % (len(SAMPLES) - 1)].replace("\x00", " ")
            store.append_message(f"session{i % 4}", "user", f"token{i % 40} {content}", event_id=f"event{i}")
        for i in range(10):
            store.propose(f"Orion.launch{i}", SAMPLES[1 + i % (len(SAMPLES) - 1)], f"event{i}", proposal_id=f"p{i}")
            store.accept(f"p{i}", authority="synthetic-host")
        exercise_ranked_calls(store)
        before = store.messages(query="Orion stock")
        with MemoryStore(path, principal_id="bob", namespace="work") as other:
            for i in range(50):
                other.append_message("foreign", "user", "Orion stock stock stock", event_id=f"foreign{i}")
        assert store.messages(query="Orion stock") == before
        store.append_message("correction", "user", "Orion launches Friday", event_id="correction")
        store.propose("Orion.launch0", "Friday", "correction", proposal_id="corrected", expected_revision=1)
        store.accept("corrected", authority="synthetic-host")
        store.forget_event("event1")
        exercise_ranked_calls(store)
        assert store.facts("Orion.launch0")[0]["value"] == "Friday"
        assert store.facts("Orion.launch1") == []
        assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    with MemoryStore(path, principal_id="alice", namespace="work") as reopened:
        exercise_ranked_calls(reopened)
        assert reopened.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_absent_fact_snapshot_can_commit_unknown_but_rejects_later_insertion(tmp_path):
    path = tmp_path / "absence.sqlite3"
    with MemoryStore(path, principal_id="alice") as store:
        store.append_message("qa", "user", "Where is Orion?", event_id="query")
        snapshot = store.fact_snapshot("Orion.location")
        assert snapshot == {"key": "Orion.location", "revision": 0, "record": None}
        store.append_message("qa", "assistant", "Unknown", event_id="answer1",
                             source_event_ids=["query"], expected_fact_revisions={snapshot["key"]: snapshot["revision"]})
        with MemoryStore(path, principal_id="alice") as other:
            other.append_message("updates", "user", "Orion is in the warehouse", event_id="source")
            other.propose("Orion.location", "warehouse", "source", proposal_id="proposal")
            other.accept("proposal", authority="synthetic-host")
        with pytest.raises(ConflictError, match="fact changed"):
            store.append_message("qa", "assistant", "Unknown", event_id="stale-answer",
                                 source_event_ids=["query"], expected_fact_revisions={snapshot["key"]: snapshot["revision"]})
        assert store.get_message("stale-answer") is None
        assert store.fact_snapshot("Orion.location")["record"]["value"] == "warehouse"


def test_deleted_fact_snapshot_tracks_recreation_and_live_dependency_erasure(tmp_path):
    path = tmp_path / "deleted.sqlite3"
    with MemoryStore(path, principal_id="alice") as store:
        store.append_message("updates", "user", "Orion is in the warehouse", event_id="source")
        store.propose("Orion.location", "warehouse", "source", proposal_id="p1")
        store.accept("p1", authority="synthetic-host")
        live = store.fact_snapshot("Orion.location")
        store.forget_event("source")
        deleted = store.fact_snapshot("Orion.location")
        assert deleted == {"key": "Orion.location", "revision": 2, "record": None}
        with pytest.raises(ConflictError, match="fact changed"):
            store.append_message("qa", "assistant", "warehouse", event_id="stale-live",
                                 expected_fact_revisions={live["key"]: live["revision"]})
        store.append_message("qa", "assistant", "Unknown", event_id="unknown",
                             expected_fact_revisions={deleted["key"]: deleted["revision"]})
        store.append_message("updates", "user", "Orion is at the port", event_id="new-source")
        store.propose("Orion.location", "port", "new-source", proposal_id="p2", expected_revision=2)
        store.accept("p2", authority="synthetic-host")
        with pytest.raises(ConflictError, match="fact changed"):
            store.append_message("qa", "assistant", "Unknown", event_id="stale-deleted",
                                 expected_fact_revisions={deleted["key"]: deleted["revision"]})
        assert store.fact_snapshot("Orion.location")["revision"] == 3


def test_fact_snapshot_distinguishes_accepted_null_from_absence_and_is_scoped(tmp_path):
    path = tmp_path / "scope.sqlite3"
    with MemoryStore(path, principal_id="alice") as store:
        store.append_message("s", "user", "The accepted value is null", event_id="source")
        store.propose("value", None, "source", proposal_id="p")
        store.accept("p", authority="synthetic-host")
        result = store.fact_snapshot("value")
        assert result["revision"] == 1 and result["record"] is not None
        assert result["record"]["value"] is None
        assert result["record"]["source_event_id"] == "source"
        with MemoryStore(path, principal_id="bob") as other:
            assert other.fact_snapshot("value") == {"key": "value", "revision": 0, "record": None}
