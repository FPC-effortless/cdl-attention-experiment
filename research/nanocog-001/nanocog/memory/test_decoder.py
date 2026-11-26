"""Persistence boundaries: actual new processes, retries, conflicts and bad state."""
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..comm.packet import PacketState
from ..interaction import PacketAndTypedProbe, PacketMixer
from ..model import Config, Decoder, DecodeState, LocalOnly
from ..state.records import RecordState
from .codec import CodecError, CodecLimits, decode, encode
from .decoder import PersistentDecoder, validate_decode_state
from .store import ConflictError, DeletedError, IntegrityError, MemoryStore, ReceiptExpiredError


def make_model(kind="local", seed=731):
    torch.manual_seed(seed)
    cls = {"local": LocalOnly, "packet": PacketMixer, "probe": PacketAndTypedProbe}[kind]
    model = Decoder(Config(31, 32, 2), cls).eval()
    with torch.no_grad():
        model.smear_gain.fill_(0.7)
    return model


def _resume_process(path, kind, tokens):
    torch.set_num_threads(1)
    with MemoryStore(path, principal_id="alice", namespace="experiment") as store:
        model = make_model(kind)
        session = PersistentDecoder(model, store, session_id="chat", tokenizer_id="fixture-tokenizer-v1")
        result = session.run(torch.tensor(tokens, dtype=torch.long), request_id="second")
        print(json.dumps({"logits": result.logits.tolist(), "revision": result.revision,
                          "tokens_seen": result.tokens_seen, "replayed": result.replayed}))


class CodecTests(unittest.TestCase):
    def test_known_nested_state_preserves_exact_tensor_bytes(self):
        packet = PacketState(torch.zeros(1, 2, 3), torch.ones(1, 1, 1, 1, 2),
                             torch.ones(1, 1, 1, 1), torch.tensor([2]))
        record = RecordState(torch.tensor([[-1, 0]]), torch.tensor([[0, 1]]),
                             torch.tensor([2]), torch.tensor([[-1, 1]]))
        value = DecodeState(((packet, record), torch.arange(3, dtype=torch.float64)),
                            torch.zeros(1, 1, 3), 2)
        restored = decode(json.loads(json.dumps(encode(value))))
        self.assertEqual(restored.tokens_seen, 2)
        for original, actual in zip(packet._parts(), restored.mixer_states[0][0]._parts()):
            self.assertTrue(torch.equal(original, actual))
            self.assertNotEqual(original.data_ptr(), actual.data_ptr())
        self.assertTrue(torch.equal(record.value, restored.mixer_states[0][1].value))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool):
            original = torch.tensor([0, 1], dtype=dtype)
            actual = decode(encode(original))
            self.assertEqual(actual.dtype, original.dtype)
            self.assertTrue(torch.equal(original, actual))
        self.assertEqual(decode(encode(torch.empty(1, 0, 3))).shape, (1, 0, 3))

    def test_unknown_types_and_executable_serialization_are_rejected(self):
        class Untrusted:
            def __reduce__(self):
                raise AssertionError("must never invoke object reduction")
        with self.assertRaises(CodecError):
            encode(Untrusted())
        wire = {"format": "nanocog-state", "version": 1,
                "value": {"type": "pickle", "data": "not executable"}}
        with self.assertRaises(CodecError):
            decode(wire)
        wire["value"] = {"type": "int", "value": True}
        with self.assertRaises(CodecError):
            decode(wire)

    def test_corruption_nonfinite_and_size_bombs_fail_closed(self):
        wire = encode(torch.ones(2))
        wire["value"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CodecError, "checksum"):
            decode(wire)
        with self.assertRaisesRegex(CodecError, "nonfinite"):
            encode(torch.tensor([float("nan")]))
        wire = encode(torch.ones(1))
        raw = bytes.fromhex("0000807f")  # float32 +inf with a consistent byte hash
        wire["value"].update(data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(CodecError, "nonfinite"):
            decode(wire)
        wire["value"]["shape"] = [0, 2**24, 2**24]
        with self.assertRaises(CodecError):
            decode(wire)
        with self.assertRaises(CodecError):
            encode(torch.ones(100), limits=CodecLimits(max_bytes=32))
        value = None
        for _ in range(30):
            value = (value,)
        with self.assertRaisesRegex(CodecError, "nesting"):
            encode(value)


class PersistentDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path, principal_id="alice", namespace="experiment")
        self.addCleanup(self.store.close)

    def session(self, model=None, session_id="chat", store=None, **kwargs):
        return PersistentDecoder(model if model is not None else make_model(), store or self.store,
                                 session_id=session_id, tokenizer_id="fixture-tokenizer-v1", **kwargs)

    def test_new_process_resume_matches_uninterrupted_decoder(self):
        prefix = torch.tensor([[2, 5, 7, 3]])
        suffix = torch.tensor([[4, 9, 11]])
        package_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=str(package_root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for kind in ("local", "packet", "probe"):
            with self.subTest(kind=kind):
                path = Path(self.temp.name) / f"{kind}.sqlite3"
                model = make_model(kind)
                with torch.inference_mode():
                    expected, full_state = model(torch.cat((prefix, suffix), dim=1))
                with MemoryStore(path, principal_id="alice", namespace="experiment") as db:
                    adapter = self.session(model, store=db)
                    first = adapter.run(prefix, request_id="first")
                    self.assertEqual(first.tokens_seen, 4)
                    self.assertFalse(first.replayed)
                    torch.testing.assert_close(first.logits, expected[:, :4], rtol=1e-5, atol=1e-6)
                child = subprocess.run(
                    [sys.executable, "-m", "nanocog.memory.test_decoder", "--resume", str(path), kind,
                     json.dumps(suffix.tolist())], cwd=package_root, env=env,
                    capture_output=True, text=True, check=True, timeout=45)
                actual = json.loads(child.stdout)
                self.assertEqual(actual["tokens_seen"], full_state.tokens_seen)
                self.assertEqual(actual["revision"], 2)
                self.assertFalse(actual["replayed"])
                torch.testing.assert_close(torch.tensor(actual["logits"]), expected[:, 4:], rtol=1e-5, atol=1e-6)

    def test_old_exact_retry_returns_original_output_without_advancing(self):
        adapter = self.session()
        first = adapter.run(torch.tensor([[1, 2]]), request_id="one")
        adapter.run(torch.tensor([[3]]), request_id="two")
        with patch.object(self.store, "load_checkpoint", side_effect=AssertionError("retry must use receipt")):
            again = adapter.run(torch.tensor([[1, 2]]), request_id="one")
        self.assertTrue(again.replayed)
        self.assertEqual((again.revision, again.tokens_seen), (1, 2))
        self.assertTrue(torch.equal(first.logits, again.logits))
        current = self.store.load_checkpoint("chat")
        self.assertEqual(current["revision"], 2)
        self.assertEqual(decode(current["payload"]["state"]).tokens_seen, 3)
        with self.assertRaises(ConflictError):
            adapter.run(torch.tensor([[2, 1]]), request_id="one")

    def test_different_session_starts_a_new_transient_stream(self):
        model = make_model()
        a, b = self.session(model, "a"), self.session(model, "b")
        a.run(torch.tensor([[1, 2, 3]]), request_id="a1")
        tokens = torch.tensor([[7, 8]])
        result = b.run(tokens, request_id="b1")
        with torch.inference_mode():
            expected, _ = model(tokens)
        self.assertEqual(result.tokens_seen, 2)
        torch.testing.assert_close(result.logits, expected)

    def test_other_principal_cannot_resume_the_same_session_name(self):
        self.session().run(torch.tensor([[1, 2]]), request_id="first")
        with MemoryStore(self.path, principal_id="bob", namespace="experiment") as db:
            self.assertIsNone(db.load_checkpoint("chat"))
            result = self.session(store=db).run(torch.tensor([[4]]), request_id="first")
            self.assertEqual(result.tokens_seen, 1)
        self.assertEqual(decode(self.store.load_checkpoint("chat")["payload"]["state"]).tokens_seen, 2)

    def test_rejects_same_geometry_wrong_model_tokenizer_and_live_mutation(self):
        model = make_model()
        adapter = self.session(model)
        adapter.run(torch.tensor([[1]]), request_id="first")
        with self.assertRaisesRegex(IntegrityError, "mismatch"):
            self.session(make_model(seed=732))
        with self.assertRaisesRegex(IntegrityError, "mismatch"):
            PersistentDecoder(make_model(), self.store, session_id="chat", tokenizer_id="other-tokenizer")
        with torch.no_grad():
            model.embedding.weight[0, 0] += 0.5
        with self.assertRaisesRegex(IntegrityError, "changed"):
            adapter.run(torch.tensor([[2]]), request_id="second")
        self.assertEqual(self.store.load_checkpoint("chat")["revision"], 1)

    def test_eval_single_stream_and_resource_limits_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "eval"):
            self.session(make_model().train())
        adapter = self.session()
        with self.assertRaises(ValueError):
            adapter.run(torch.tensor([[1], [2]]), request_id="batch")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                adapter.run(torch.tensor([[1]]), request_id="cast")
        small = self.session(session_id="small", codec_limits=CodecLimits(max_bytes=64))
        with self.assertRaisesRegex(CodecError, "output"):
            small.run(torch.tensor([[1]]), request_id="oversize")
        self.assertIsNone(self.store.load_checkpoint("small"))

    def test_bad_typed_chronology_and_packet_payload_are_rejected(self):
        model = make_model("probe")
        with torch.inference_mode():
            _, state = model(torch.tensor([[1, 2]]))
        packet, record = state.mixer_states[0]
        bad_record = replace(record, clock=record.clock + 10)
        bad = replace(state, mixer_states=((packet, bad_record), *state.mixer_states[1:]))
        with self.assertRaisesRegex(IntegrityError, "clock"):
            validate_decode_state(bad, model)
        bad_packet = replace(packet, present=torch.full_like(packet.present, 0.5))
        bad = replace(state, mixer_states=((bad_packet, record), *state.mixer_states[1:]))
        with self.assertRaisesRegex(IntegrityError, "binary"):
            validate_decode_state(bad, model)
        bad = replace(state, mixer_states=((packet, replace(record, value=42)), *state.mixer_states[1:]))
        with self.assertRaises(IntegrityError):
            validate_decode_state(bad, model)

    def test_semantically_invalid_but_checksummed_checkpoint_is_rejected(self):
        model = make_model("probe")
        adapter = self.session(model)
        with torch.inference_mode():
            _, state = model(torch.tensor([[1, 2]]))
        packet, record = state.mixer_states[0]
        bad = replace(state, mixer_states=((packet, replace(record, clock=record.clock + 3)),
                                          *state.mixer_states[1:]))
        self.store.commit_checkpoint("chat", request_id="forged", request_hash="0" * 64,
                                     expected_revision=0, manifest=adapter._manifest,
                                     payload={"state": encode(bad)}, result={})
        with self.assertRaises(IntegrityError):
            self.session(make_model("probe"))
        with self.assertRaises(IntegrityError):
            adapter.run(torch.tensor([[3]]), request_id="next")
        self.assertEqual(self.store.load_checkpoint("chat")["revision"], 1)

    def test_nonfinite_computation_does_not_commit(self):
        model = make_model()
        with torch.no_grad():
            model.mixers[0].conv.weight.fill_(1e38)
            model.mixers[0].out.weight.fill_(1e38)
        adapter = self.session(model)
        with self.assertRaises((IntegrityError, CodecError)):
            adapter.run(torch.tensor([[1, 2]]), request_id="overflow")
        self.assertIsNone(self.store.load_checkpoint("chat"))
        self.assertIsNone(self.store.checkpoint_receipt("chat", "overflow"))

    def test_stale_writer_loses_without_returning_uncommitted_result(self):
        first = self.session()
        with MemoryStore(self.path, principal_id="alice", namespace="experiment") as other:
            second = self.session(store=other)
            original = self.store.commit_checkpoint
            def race(*args, **kwargs):
                second.run(torch.tensor([[8]]), request_id="winner")
                return original(*args, **kwargs)
            with patch.object(self.store, "commit_checkpoint", side_effect=race):
                with self.assertRaises(ConflictError):
                    first.run(torch.tensor([[2]]), request_id="loser")
        self.assertIsNone(self.store.checkpoint_receipt("chat", "loser"))
        result = first.run(torch.tensor([[2]]), request_id="loser")
        with torch.inference_mode():
            expected, _ = first.model(torch.tensor([[8, 2]]))
        self.assertEqual((result.revision, result.tokens_seen), (2, 2))
        torch.testing.assert_close(result.logits, expected[:, -1:], rtol=1e-5, atol=1e-6)

    def test_simultaneous_identical_request_commits_once(self):
        first = self.session()
        with MemoryStore(self.path, principal_id="alice", namespace="experiment") as other:
            second = self.session(store=other)
            original = self.store.commit_checkpoint
            def race(*args, **kwargs):
                second.run(torch.tensor([[2]]), request_id="same")
                return original(*args, **kwargs)
            with patch.object(self.store, "commit_checkpoint", side_effect=race):
                result = first.run(torch.tensor([[2]]), request_id="same")
        self.assertTrue(result.replayed)
        self.assertEqual((result.revision, result.tokens_seen), (1, 1))
        self.assertEqual(self.store.load_checkpoint("chat")["revision"], 1)

    def test_expired_receipt_never_reexecutes_committed_input(self):
        adapter = self.session()
        with patch("nanocog.memory.store.CHECKPOINT_OUTPUTS", 2):
            for i in range(3):
                adapter.run(torch.tensor([[i + 1]]), request_id=str(i))
        with self.assertRaises(ReceiptExpiredError):
            adapter.run(torch.tensor([[1]]), request_id="0")
        self.assertEqual(self.store.load_checkpoint("chat")["revision"], 3)
        self.assertEqual(decode(self.store.load_checkpoint("chat")["payload"]["state"]).tokens_seen, 3)

    def test_erasure_invalidates_live_adapter_and_old_receipts(self):
        adapter = self.session()
        adapter.run(torch.tensor([[1]]), request_id="one")
        self.store.forget_session("chat")
        with self.assertRaises((DeletedError, ReceiptExpiredError)):
            adapter.run(torch.tensor([[1]]), request_id="one")
        with self.assertRaises(DeletedError):
            adapter.run(torch.tensor([[2]]), request_id="two")
        with self.assertRaises(DeletedError):
            self.session()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--resume":
        _resume_process(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
    else:
        unittest.main()
