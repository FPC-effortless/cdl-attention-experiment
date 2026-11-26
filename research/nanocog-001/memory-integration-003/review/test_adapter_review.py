"""Independent fault fixtures; these tests measure no learned semantic accuracy."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import string
import sys

import pytest
import torch
from torch import nn

RUNTIME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME))
from nanocog.memory.store import ConflictError, IntegrityError, MemoryStore

if os.environ.get("REVIEW_ADAPTER_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "nanocog.memory.independent_review_adapter", os.environ["REVIEW_ADAPTER_SOURCE"])
    adapter_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter_module
    spec.loader.exec_module(adapter_module)
else:
    from nanocog.memory import learned_archive as adapter_module

torch.set_num_threads(1)


@dataclass
class MiniConfig:
    architecture_id: str = "cog-unified-connected-v2"
    d_model: int = 122
    vocab_size: int = len(string.printable)


class MiniBoard(nn.Module):
    use_scripted_core = False

    def forward(self, features, memory, controller, **kwargs):
        return features, memory, controller, None


class MiniNative(nn.Module):
    """Only an interface fixture, explicitly not the admitted trained backbone."""
    def __init__(self):
        super().__init__()
        self.config = MiniConfig()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.default_feedback_enabled = True
        self.record_board = MiniBoard()
        self.final_norm = nn.Identity()

    def init_state(self, batch, device):
        return {"memory": None, "controller": None}

    def backbone_features(self, ids, state):
        return torch.zeros(*ids.shape, 122, device=ids.device), state


class CallbackAdapter(adapter_module.FactualAdapter):
    def __init__(self, callback=None):
        super().__init__(MiniNative(), string.printable)
        self.callback = callback
        with torch.no_grad():
            self.writer.weight.zero_()
            self.writer.bias.zero_()
            self.writer.bias[adapter_module.operation(0, 0)] = 10
            self.query.weight.zero_()
            self.query.bias.zero_()
            self.query.bias[0] = 10
            self.answer.weight.zero_()
            self.answer.bias.zero_()
            for owner in range(8):
                self.answer.weight[owner, owner] = 10
            self.answer.weight[8, 12] = 10

    def predict_events(self, texts):
        if self.callback:
            self.callback()
        return super().predict_events(texts)


def accept_owner(store, owner, token):
    key = adapter_module.fact_key(0)
    store.append_message("updates", "user", f"owner update {token}", event_id=token)
    store.propose(key, adapter_module.fact_payload(0, owner), token,
                  proposal_id="proposal-" + token, expected_revision=store.fact_revision(key))
    return store.accept("proposal-" + token, authority="review-host")


def test_global_forward_hook_cannot_rewrite_a_factual_decision(tmp_path):
    adapter = CallbackAdapter()
    with MemoryStore(tmp_path / "hook.sqlite3", principal_id="review") as store:
        memory = adapter_module.FactualMemory(store, adapter, session_id="chat", authority="review-host")
        assert adapter.predict_events(["Amara holds the chest."]) == [adapter_module.operation(0, 0)]

        def corrupt_writer(module, args, output):
            if module is adapter.writer:
                changed = torch.zeros_like(output)
                changed[:, adapter_module.operation(0, 1)] = 100
                return changed

        handle = torch.nn.modules.module.register_module_forward_hook(corrupt_writer)
        try:
            with pytest.raises((IntegrityError, ValueError), match="model|hook"):
                memory.observe("Amara holds the chest.", request_id="event")
        finally:
            handle.remove()
        assert store.facts(adapter_module.fact_key(0)) == []


@pytest.mark.parametrize("initial_owner", [None, 6])
def test_write_does_not_rebase_onto_a_change_during_prediction(tmp_path, initial_owner):
    path = tmp_path / "writer-race.sqlite3"
    with MemoryStore(path, principal_id="review") as store, \
         MemoryStore(path, principal_id="review") as concurrent:
        if initial_owner is not None:
            accept_owner(store, initial_owner, "initial")
        adapter = CallbackAdapter(lambda: accept_owner(concurrent, 1, "intervening"))
        memory = adapter_module.FactualMemory(store, adapter, session_id="chat", authority="review-host")
        with pytest.raises(ConflictError, match="revision"):
            memory.observe("Amara holds the chest.", request_id="older-event")
        current = store.facts(adapter_module.fact_key(0))[0]
        assert current["value"]["value"] == "Bela"
        assert current["revision"] == (1 if initial_owner is None else 2)
