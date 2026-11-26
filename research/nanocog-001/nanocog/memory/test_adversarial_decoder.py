"""Independent durability adversaries for the reference inference adapter.

These fixtures use seeded, untrained weights. Passing them is an engineering
claim about state/identity boundaries, not learned language or memory accuracy.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..interaction import PacketAndTypedProbe, PacketMixer
from ..model import Config, Decoder, LocalOnly
from .codec import CodecError, CodecLimits, decode, encode
from .decoder import PersistentDecoder
from .store import ConflictError, DeletedError, IntegrityError, MemoryStore, ReceiptExpiredError


def fixture_model(kind="local", seed=731, dtype=torch.float32):
    torch.manual_seed(seed)
    cls = {"local": LocalOnly, "packet": PacketMixer, "probe": PacketAndTypedProbe}[kind]
    model = Decoder(Config(31, 32, 2), cls).to(dtype=dtype).eval()
    with torch.no_grad():
        model.smear_gain.fill_(0.7)
    return model


class AdversarialDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "memory.db",
                                 principal_id="alice", namespace="independent-validation")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def adapter(self, model=None, session="chat", **kwargs):
        return PersistentDecoder(fixture_model() if model is None else model, self.store,
                                 session_id=session, tokenizer_id="synthetic-vocab31-v1", **kwargs)

    def source(self):
        self.store.append_message("source-session", "user", "erasure race source",
                                  event_id="source")

    def test_changed_convolution_execution_geometry_rejected_live(self):
        model = fixture_model()
        adapter = self.adapter(model)
        adapter.run(torch.tensor([[2, 3, 4]]), request_id="first")
        # Output geometry and all tensor weights remain identical. The computation
        # changes: convolution samples a different history with new padding.
        model.mixers[0].conv.padding = (1,)
        model.mixers[0].conv.dilation = (2,)
        with self.assertRaises((IntegrityError, ValueError)):
            adapter.run(torch.tensor([[5, 6]]), request_id="changed")
        self.assertEqual(self.store.load_checkpoint("chat")["revision"], 1)

    def test_changed_convolution_execution_geometry_rejected_on_reopen(self):
        self.adapter().run(torch.tensor([[2, 3, 4]]), request_id="first")
        different = fixture_model()
        different.mixers[0].conv.padding = (1,)
        different.mixers[0].conv.dilation = (2,)
        with self.assertRaises((IntegrityError, ValueError)):
            self.adapter(different)

    def test_global_forward_hook_cannot_change_persisted_inference(self):
        model = fixture_model()
        adapter = self.adapter(model)

        def alter(module, _inputs, output):
            if module is model.output:
                return output * 0.5

        handle = torch.nn.modules.module.register_module_forward_hook(alter)
        try:
            with self.assertRaises((IntegrityError, ValueError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="hooked")
        finally:
            handle.remove()
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_global_pre_hook_cannot_substitute_the_requested_tokens(self):
        model = fixture_model()
        adapter = self.adapter(model)

        def substitute(module, inputs):
            if module is model:
                return (torch.full_like(inputs[0], 17), *inputs[1:])

        handle = torch.nn.modules.module.register_module_forward_pre_hook(substitute)
        try:
            with self.assertRaises((IntegrityError, ValueError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="substituted")
        finally:
            handle.remove()
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_class_level_local_forward_replacement_is_rejected(self):
        adapter = self.adapter()
        original = LocalOnly.forward

        def replacement(module, x, state=None):
            output, state = original(module, x, state)
            return output * 0.5, state

        with patch.object(LocalOnly, "forward", replacement):
            with self.assertRaises((IntegrityError, ValueError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="patched")
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_class_level_torch_forward_replacement_is_rejected(self):
        adapter = self.adapter()
        original = torch.nn.Linear.forward

        def replacement(module, x):
            return original(module, x) * 0.5

        with patch.object(torch.nn.Linear, "forward", replacement):
            with self.assertRaises((IntegrityError, ValueError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="patched")
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_compiled_call_replacement_is_rejected(self):
        model = fixture_model()
        adapter = self.adapter(model)
        model._compiled_call_impl = lambda tokens, state=None: model.forward(tokens, state)
        with self.assertRaises((IntegrityError, ValueError)):
            adapter.run(torch.tensor([[2, 3, 4]]), request_id="compiled")
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_first_checkpoint_cannot_resurrect_state_after_source_erasure(self):
        self.source()
        adapter = self.adapter()
        original = self.store.commit_checkpoint

        def erase_then_commit(*args, **kwargs):
            self.store.forget_event("source")
            return original(*args, **kwargs)

        with patch.object(self.store, "commit_checkpoint", side_effect=erase_then_commit):
            with self.assertRaises((DeletedError, ConflictError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="first")
        rows = self.store.db.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
        self.assertEqual(rows, 0)

    def test_existing_checkpoint_cannot_resurrect_state_after_source_erasure(self):
        self.source()
        adapter = self.adapter()
        adapter.run(torch.tensor([[2, 3, 4]]), request_id="first")
        original = self.store.commit_checkpoint

        def erase_then_commit(*args, **kwargs):
            self.store.forget_event("source")
            return original(*args, **kwargs)

        with patch.object(self.store, "commit_checkpoint", side_effect=erase_then_commit):
            with self.assertRaises((DeletedError, ConflictError)):
                adapter.run(torch.tensor([[5, 6]]), request_id="second")
        rows = self.store.db.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
        self.assertEqual(rows, 0)

    def test_deleted_session_blocks_a_first_checkpoint_in_flight(self):
        adapter = self.adapter()
        original = self.store.commit_checkpoint

        def erase_then_commit(*args, **kwargs):
            self.store.forget_session("chat")
            return original(*args, **kwargs)

        with patch.object(self.store, "commit_checkpoint", side_effect=erase_then_commit):
            with self.assertRaises((DeletedError, ConflictError)):
                adapter.run(torch.tensor([[2, 3, 4]]), request_id="first")

    def test_corrupt_receipt_revision_cannot_misreport_an_old_retry(self):
        adapter = self.adapter()
        tokens = torch.tensor([[2, 3, 4]])
        adapter.run(tokens, request_id="first")
        adapter.run(torch.tensor([[5, 6]]), request_id="second")
        self.store.db.execute("UPDATE checkpoint_requests SET revision=101 WHERE request='first'")
        with self.assertRaises(IntegrityError):
            adapter.run(tokens, request_id="first")

    def test_corrupt_receipt_payload_is_rejected(self):
        adapter = self.adapter()
        tokens = torch.tensor([[2, 3, 4]])
        adapter.run(tokens, request_id="first")
        self.store.db.execute("UPDATE checkpoint_requests SET result='{}' WHERE request='first'")
        with self.assertRaises(IntegrityError):
            adapter.run(tokens, request_id="first")

    def test_retry_snapshot_preceding_erasure_is_not_revoked_after_read(self):
        # A receipt read is the linearization point. This deliberately records
        # the boundary: deletion cannot retract bytes an overlapping call read.
        self.source()
        adapter = self.adapter()
        tokens = torch.tensor([[2, 3, 4]])
        original_result = adapter.run(tokens, request_id="first")
        original_read = self.store.checkpoint_receipt

        def read_then_erase(*args, **kwargs):
            receipt = original_read(*args, **kwargs)
            self.store.forget_event("source")
            return receipt

        with patch.object(self.store, "checkpoint_receipt", side_effect=read_then_erase):
            replay = adapter.run(tokens, request_id="first")
        self.assertTrue(replay.replayed)
        torch.testing.assert_close(replay.logits, original_result.logits, rtol=0, atol=0)
        with self.assertRaises(ReceiptExpiredError):
            adapter.run(tokens, request_id="first")

    def test_returned_logits_do_not_alias_the_saved_receipt(self):
        adapter = self.adapter()
        tokens = torch.tensor([[2, 3, 4]])
        result = adapter.run(tokens, request_id="first")
        correct = result.logits.clone()
        result.logits.fill_(0)
        replay = adapter.run(tokens, request_id="first")
        torch.testing.assert_close(replay.logits, correct, rtol=0, atol=0)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.tokens_seen, 3)

    def test_caller_token_mutation_after_inference_does_not_change_request_identity(self):
        adapter = self.adapter()
        original_tokens = torch.tensor([[2, 3, 4]])
        caller_tokens = original_tokens.clone()
        original = self.store.commit_checkpoint

        def mutate_caller(*args, **kwargs):
            caller_tokens.fill_(17)
            return original(*args, **kwargs)

        with patch.object(self.store, "commit_checkpoint", side_effect=mutate_caller):
            result = adapter.run(caller_tokens, request_id="first")
        replay = adapter.run(original_tokens, request_id="first")
        torch.testing.assert_close(result.logits, replay.logits, rtol=0, atol=0)
        with self.assertRaises(ConflictError):
            adapter.run(caller_tokens, request_id="first")

    def test_noncontiguous_valid_token_views_have_the_same_result(self):
        tokens = torch.arange(14, dtype=torch.long).reshape(1, 14)[:, ::2]
        self.assertFalse(tokens.is_contiguous())
        result = self.adapter(session="view").run(tokens, request_id="first")
        control = self.adapter(session="copy").run(tokens.contiguous(), request_id="first")
        torch.testing.assert_close(result.logits, control.logits, rtol=0, atol=0)

    def test_invalid_input_bounds_never_create_a_checkpoint(self):
        adapter = self.adapter(max_tokens_per_run=8)
        cases = [torch.tensor([1, 2]), torch.empty((1, 0), dtype=torch.long),
                 torch.empty((0, 2), dtype=torch.long), torch.ones((2, 2), dtype=torch.long),
                 torch.ones((1, 2), dtype=torch.float32), torch.ones((1, 2), dtype=torch.int32),
                 torch.tensor([[-1]]), torch.tensor([[31]]), torch.ones((1, 9), dtype=torch.long)]
        for i, tokens in enumerate(cases):
            with self.subTest(case=i):
                with self.assertRaises((TypeError, ValueError)):
                    adapter.run(tokens, request_id=f"bad-{i}")
                self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_sparse_tokens_are_rejected_as_an_input_boundary_error(self):
        tokens = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 1]]),
                                         torch.tensor([2, 3]), (1, 2))
        with self.assertRaises(ValueError):
            self.adapter().run(tokens, request_id="sparse")
        self.assertIsNone(self.store.load_checkpoint("chat"))

    def test_float64_continuation_for_all_three_supported_mixers(self):
        tokens = torch.tensor([[2, 3, 4, 7, 11, 5, 9, 13, 1]])
        for kind in ("local", "packet", "probe"):
            with self.subTest(kind=kind):
                model = fixture_model(kind, dtype=torch.float64)
                with torch.inference_mode():
                    expected, _ = model(tokens)
                self.adapter(model, session=kind).run(tokens[:, :2], request_id="first")
                resumed = self.adapter(fixture_model(kind, dtype=torch.float64), session=kind)
                got = resumed.run(tokens[:, 2:], request_id="second")
                torch.testing.assert_close(got.logits, expected[:, 2:], rtol=1e-6, atol=1e-7)
                self.assertEqual(got.tokens_seen, tokens.numel())


class AdversarialCodecTests(unittest.TestCase):
    def test_strided_tensor_values_roundtrip_without_storage_aliases(self):
        source = torch.arange(60, dtype=torch.int16).reshape(6, 10).transpose(0, 1)[::2]
        encoded = encode(source)
        restored = decode(encoded)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)
        restored.fill_(-7)
        self.assertFalse(torch.equal(source, restored))
        torch.testing.assert_close(decode(encoded), source, rtol=0, atol=0)

    def test_invalid_boolean_bytes_rejected_even_with_matching_hash(self):
        import base64
        import hashlib
        document = encode(torch.tensor([True], dtype=torch.bool))
        document["value"]["data"] = base64.b64encode(b"\x02").decode()
        document["value"]["sha256"] = hashlib.sha256(b"\x02").hexdigest()
        with self.assertRaisesRegex(CodecError, "Boolean"):
            decode(document)

    def test_noncanonical_base64_rejected_even_with_matching_decoded_bytes(self):
        document = encode(torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(document["value"]["data"], "AQ==")
        document["value"]["data"] = "AR=="
        with self.assertRaisesRegex(CodecError, "canonical"):
            decode(document)

    def test_schema_extensions_are_not_silently_deserialized(self):
        for path in ("envelope", "tensor"):
            with self.subTest(path=path):
                document = encode(torch.ones(2))
                target = document if path == "envelope" else document["value"]
                target["extra"] = "unrecognized"
                with self.assertRaises(CodecError):
                    decode(document)

    def test_aggregate_tensor_budget_not_just_per_tensor(self):
        document = encode((torch.zeros(5), torch.zeros(5)))
        with self.assertRaisesRegex(CodecError, "byte/count"):
            decode(document, limits=CodecLimits(max_bytes=32))


if __name__ == "__main__":
    unittest.main()
