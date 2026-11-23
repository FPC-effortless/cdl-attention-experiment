"""Serving optimizations must preserve byte identity and live mutation guards."""
import struct
from unittest.mock import patch

import pytest
import torch
from torch import nn

from ..model import Config, Decoder, LocalOnly
from .codec import CodecError, _tensor_buffer, tensor_bytes
from .decoder import PersistentDecoder, _model_identity
from .store import IntegrityError, MemoryStore


def model():
    torch.manual_seed(91023)
    return Decoder(Config(31, 32, 2), LocalOnly).eval()


def runner(db, net):
    return PersistentDecoder(net, db, session_id="serving", tokenizer_id="fixture-v1")


@pytest.mark.parametrize("dtype,values,fmt", [
    (torch.float32, [-1.5, 0.0, 2.25], "<fff"),
    (torch.float64, [-1.5, 0.0, 2.25], "<ddd"),
    (torch.int64, [-1, 0, 2], "<qqq"),
    (torch.int16, [-1, 0, 2], "<hhh"),
    (torch.bool, [False, True, False], "<???"),
])
def test_canonical_hash_view_matches_explicit_little_endian_bytes(dtype, values, fmt):
    tensor = torch.tensor(values, dtype=dtype)
    expected = struct.pack(fmt, *values)
    view = _tensor_buffer(tensor)
    assert view.contiguous and view.format == "B"
    assert bytes(view) == tensor_bytes(tensor) == expected


def test_noncontiguous_hash_view_preserves_logical_tensor_order_and_lifetime():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t()
    assert not source.is_contiguous()
    view = _tensor_buffer(source)
    del source
    assert bytes(view) == struct.pack("<ffff", 1.0, 3.0, 2.0, 4.0)
    assert bytes(_tensor_buffer(torch.empty(1, 0, 3))) == b""


@pytest.mark.parametrize("value", [0.5, float("inf"), float("nan")])
def test_data_mutation_is_rejected_without_relying_on_version_counter(tmp_path, value):
    net = model()
    with MemoryStore(tmp_path / "memory.sqlite3", principal_id="test") as db:
        session = runner(db, net)
        session.run(torch.tensor([[1, 2]]), request_id="first")
        version = net.embedding.weight._version
        if value == 0.5:
            net.embedding.weight.data[0, 0] += value
        else:
            net.embedding.weight.data[0, 0] = value
        assert net.embedding.weight._version == version
        with pytest.raises(IntegrityError, match="changed"):
            session.run(torch.tensor([[3]]), request_id="next")
        assert db.load_checkpoint("serving")["revision"] == 1


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_initial_model_finiteness_is_still_checked(tmp_path, value):
    net = model()
    net.embedding.weight.data[0, 0] = value
    with MemoryStore(tmp_path / "memory.sqlite3", principal_id="test") as db:
        with pytest.raises(CodecError, match="nonfinite"):
            runner(db, net)
        assert db.load_checkpoint("serving") is None


def test_post_inference_full_hash_still_rejects_live_data_mutation(tmp_path):
    net = model()
    with MemoryStore(tmp_path / "memory.sqlite3", principal_id="test") as db:
        session = runner(db, net)
        session.run(torch.tensor([[1, 2]]), request_id="first")
        original = torch.tanh

        def mutate_after_computation(x):
            answer = original(x)
            net.embedding.weight.data[0, 0] += 0.5
            return answer

        with patch.object(torch, "tanh", side_effect=mutate_after_computation):
            with pytest.raises(IntegrityError, match="changed"):
                session.run(torch.tensor([[3]]), request_id="next")
        assert db.load_checkpoint("serving")["revision"] == 1


def test_live_class_descriptor_changes_remain_in_execution_identity(tmp_path):
    net = model()
    with MemoryStore(tmp_path / "memory.sqlite3", principal_id="test") as db:
        session = runner(db, net)
        session.run(torch.tensor([[1, 2]]), request_id="first")
        # A data descriptor overrides the existing instance dictionary value.
        # Looking up only vars(module) would silently miss this change.
        with patch.object(nn.Conv1d, "padding", property(lambda self: (1,)), create=True):
            with pytest.raises(IntegrityError, match="changed"):
                session.run(torch.tensor([[3]]), request_id="next")
        assert db.load_checkpoint("serving")["revision"] == 1


def test_new_class_attribute_is_not_hidden_by_presence_optimization(tmp_path):
    net = model()
    with MemoryStore(tmp_path / "memory.sqlite3", principal_id="test") as db:
        session = runner(db, net)
        session.run(torch.tensor([[1, 2]]), request_id="first")
        with patch.object(nn.ModuleList, "topics", 3, create=True):
            with pytest.raises(IntegrityError, match="changed"):
                session.run(torch.tensor([[3]]), request_id="next")
        assert db.load_checkpoint("serving")["revision"] == 1


def test_content_hash_recheck_keeps_identical_finite_manifest():
    net = model()
    assert _model_identity(net, "fixture-v1", "s") == _model_identity(net, "fixture-v1", "s", check_finite=False)
