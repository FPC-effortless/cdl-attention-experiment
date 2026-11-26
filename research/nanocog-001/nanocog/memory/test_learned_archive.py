"""Infrastructure unit tests; fixture predictions are NOT learned-model evidence.

Real checkpoint training, inference, interventions, and restart evidence are
separate in memory-integration-003/learned. These tests isolate durable semantics.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from .learned_archive import (FactualMemory, UNKNOWN, execute_operation, fact_key,
                              fact_payload, operation, read_ledger)
from .store import ConflictError, DeletedError, IntegrityError, MemoryStore


class FixtureAdapter(nn.Module):
    def __init__(self, op=0, route=0):
        super().__init__()
        self.native = nn.Linear(1, 1)
        self.op, self.route = op, route
        self.during_event = self.during_answer = None

    def model_identity(self):
        return hashlib.sha256(self.native.weight.detach().numpy().tobytes()).hexdigest()

    def predict_events(self, texts):
        if self.during_event:
            self.during_event()
        return [self.op] * len(texts)

    def predict_queries(self, texts):
        return [self.route] * len(texts)

    def predict_answers(self, values):
        if self.during_answer:
            self.during_answer()
        return [v if v < 8 else 8 for v in values]


class LearnedArchiveInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path, principal_id="alice")
        self.adapter = FixtureAdapter()
        self.memory = FactualMemory(self.store, self.adapter, session_id="chat-a", authority="test-policy")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def accepted(self, record, value, identity):
        self.store.append_message("corrections", "user", identity, event_id=identity)
        rev = self.store.fact_snapshot(fact_key(record))["revision"]
        self.store.propose(fact_key(record), fact_payload(record, value), identity,
                           proposal_id="proposal:" + identity, expected_revision=rev)
        return self.store.accept("proposal:" + identity, authority="human")

    def test_invalid_write_is_rejected_not_repaired(self):
        self.adapter.op = operation(0, 10)
        result = self.memory.observe("some event", request_id="invalid")
        self.assertEqual(result.status, "rejected_invalid")
        self.assertEqual(self.store.facts(), [])
        ledger = [UNKNOWN] * 8
        self.assertEqual(execute_operation(ledger, self.adapter.op), "rejected_invalid")
        self.assertEqual(ledger, [UNKNOWN] * 8)

    def test_retry_does_not_reapply_old_accepted_write(self):
        self.adapter.op = operation(0, 1)
        first = self.memory.observe("Bela received it", request_id="a")
        self.accepted(0, 2, "correction")
        replay = self.memory.observe("Bela received it", request_id="a")
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.fact["revision"], first.fact["revision"])
        self.assertEqual(self.store.facts(fact_key(0))[0]["value"], fact_payload(0, 2))

    def test_retry_authority_cannot_be_rewritten(self):
        self.adapter.op = operation(0, 1)
        self.memory.observe("event", request_id="a")
        other = FactualMemory(self.store, self.adapter, session_id="chat-a", authority="different")
        with self.assertRaises(ConflictError):
            other.observe("event", request_id="a")

    def test_failed_accept_retries_same_pending_decision(self):
        self.adapter.op = operation(0, 1)
        original = self.store.accept
        def fail(*args, **kwargs):
            raise RuntimeError("interrupted before acceptance")
        self.store.accept = fail
        with self.assertRaises(RuntimeError):
            self.memory.observe("event", request_id="a")
        self.store.accept = original
        self.adapter.op = operation(1, 2)
        result = self.memory.observe("event", request_id="a")
        self.assertTrue(result.replayed)
        self.assertEqual(result.operation, operation(0, 1))
        self.assertEqual(self.store.facts(fact_key(0))[0]["value"], fact_payload(0, 1))

    def test_stale_pending_decision_does_not_rebase(self):
        self.adapter.op = operation(0, 1)
        original = self.store.accept
        self.store.accept = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("crash"))
        with self.assertRaises(RuntimeError):
            self.memory.observe("event", request_id="a")
        self.store.accept = original
        self.accepted(0, 2, "intervening")
        with self.assertRaises(ConflictError):
            self.memory.observe("event", request_id="a")

    def test_relevant_and_irrelevant_fact_interventions(self):
        self.accepted(0, 0, "owner")
        self.accepted(4, 8, "link")
        self.adapter.route = 4
        answer = self.memory.ask("question", request_id="q1")
        self.assertEqual(answer.answer, "Amara")
        self.assertEqual(set(answer.dependencies), {fact_key(0), fact_key(4)})
        self.accepted(1, 7, "irrelevant")
        self.assertEqual(self.memory.ask("question", request_id="q2").answer, "Amara")
        self.accepted(0, 2, "relevant")
        self.assertEqual(self.memory.ask("question", request_id="q3").answer, "Cato")
        self.accepted(4, 9, "link-correction")
        self.assertEqual(self.memory.ask("question", request_id="q4").answer, "Hana")

    def test_old_request_replays_historical_receipt_new_request_sees_correction(self):
        self.accepted(0, 0, "before")
        self.memory.ask("question", request_id="q")
        self.accepted(0, 1, "after")
        old = self.memory.ask("question", request_id="q")
        self.assertTrue(old.replayed)
        self.assertEqual(old.answer, "Amara")
        self.assertEqual(self.memory.ask("question", request_id="new").answer, "Bela")

    def test_relevant_edit_during_answer_rejects_commit(self):
        self.accepted(0, 0, "before")
        self.adapter.during_answer = lambda: self.accepted(0, 1, "racing")
        with self.assertRaises(ConflictError):
            self.memory.ask("question", request_id="q")
        self.assertFalse(any(m["role"] == "assistant" for m in self.store.messages("chat-a")))

    def test_irrelevant_edit_during_answer_does_not_block(self):
        self.accepted(0, 0, "before")
        self.adapter.during_answer = lambda: self.accepted(1, 1, "irrelevant-race")
        self.assertEqual(self.memory.ask("question", request_id="q").answer, "Amara")

    def test_absence_and_recreation_during_answer_reject(self):
        self.adapter.during_answer = lambda: self.accepted(0, 1, "first-insertion")
        with self.assertRaises(ConflictError):
            self.memory.ask("question", request_id="absent")
        self.store.forget_event("first-insertion")
        self.adapter.during_answer = lambda: self.accepted(0, 2, "recreation")
        with self.assertRaises(ConflictError):
            self.memory.ask("question", request_id="deleted")

    def test_stable_absence_returns_unknown_with_revision_guard(self):
        result = self.memory.ask("question", request_id="missing")
        self.assertEqual(result.answer, "unknown")
        self.assertEqual(result.dependencies, {fact_key(0): 0})

    def test_source_erasure_propagates_through_decision_and_answer(self):
        self.adapter.op = operation(0, 1)
        write = self.memory.observe("event", request_id="a")
        response = self.memory.ask("question", request_id="q")
        raw_id, _, _ = self.memory._ids("a", "observe")
        self.store.forget_event(raw_id)
        self.assertEqual(self.store.facts(), [])
        for event_id in (write.decision_event_id, response.assistant_event_id):
            with self.assertRaises(DeletedError):
                self.store.get_message(event_id)

    def test_mutation_before_replay_rejects(self):
        self.memory.ask("question", request_id="q")
        self.adapter.native.weight.data.add_(1)
        with self.assertRaises(IntegrityError):
            self.memory.ask("question", request_id="q")

    def test_mutation_during_prediction_rejects_before_result_saved(self):
        self.adapter.during_event = lambda: self.adapter.native.weight.data.add_(1)
        with self.assertRaises(IntegrityError):
            self.memory.observe("event", request_id="a")
        _, decision_id, _ = self.memory._ids("a", "observe")
        self.assertIsNone(self.store.get_message(decision_id))

    def test_writer_cannot_adopt_revision_changed_during_prediction(self):
        self.adapter.op = operation(0, 0)
        self.adapter.during_event = lambda: self.accepted(0, 1, "newer-completed-write")
        with self.assertRaises(ConflictError):
            self.memory.observe("older in-flight event", request_id="a")
        self.assertEqual(self.store.facts(fact_key(0))[0]["value"], fact_payload(0, 1))

    def test_global_forward_hooks_are_rejected(self):
        handle = torch.nn.modules.module.register_module_forward_hook(lambda module, args, output: output)
        try:
            with self.assertRaises(IntegrityError):
                self.memory.ask("question", request_id="q")
        finally:
            handle.remove()

    def test_global_pre_forward_hooks_are_rejected(self):
        handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda module, args: args)
        try:
            with self.assertRaises(IntegrityError):
                self.memory.observe("event", request_id="a")
        finally:
            handle.remove()

    def test_mutation_during_answer_rejects_before_receipt_saved(self):
        self.accepted(0, 1, "owner")
        self.adapter.during_answer = lambda: self.adapter.native.weight.data.add_(1)
        with self.assertRaises(IntegrityError):
            self.memory.ask("question", request_id="q")
        _, response_id, _ = self.memory._ids("q", "ask")
        self.assertIsNone(self.store.get_message(response_id))

    def test_query_traversal_has_no_oracle_fallback(self):
        self.accepted(0, 1, "owner")
        self.adapter.route = 4  # Missing item link cannot guess the known owner.
        self.assertEqual(self.memory.ask("question", request_id="q").answer, "unknown")
        self.assertEqual(read_ledger([1, UNKNOWN, UNKNOWN, UNKNOWN, 8, UNKNOWN, UNKNOWN, UNKNOWN], 4), 1)


if __name__ == "__main__":
    unittest.main()
