"""Thin, auditable adapters for the existing LRC model and training families.

The imported source is retained verbatim under ``vendor``.  Widths and local
kernel size below define this parameter-matched benchmark, not the historical
pilot configurations.  ``make_stream`` exposes a batch-one next-byte service:
``step(int)`` and ``prefill(list[int])`` return [1, 257] log probabilities.

There is no inferred hard-routing gradient.  Every hard-route fit calls its
original counterfactual training procedure, and the lexical fits call their
original auxiliary-supervision procedure.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parent / "vendor"
# The historical modules import their neighbours by top-level module name.
# Prefer this immutable snapshot over a checkout's newer/default branch files.
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from relational_composition.model import Config, RelationalLM
from relational_composition.expressive_model import SerialMultiHopLM, StreamingMultiHop
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop
from relational_composition.fast_inference import StreamingInference
from relational_composition.train import counterfactual_step
from relational_composition.train_hierarchical import hierarchical_credit_step
from backbone_models import Model as BackboneModel, Stream as BackboneStream, cache_batch
from train_aux import credit_step as lexical_credit_step
from utility.gated_model import GatedMultiHopLM, StreamingGatedMultiHop, LocalOnlyStream


ARMS = (
    "v1_coupled", "v1_independent", "v1_no_reference", "v3", "v4_p16", "v4",
    "lrc_local", "lrc_cache", "v4_lexical", "v4_lexical_slot", "v4_shuffled",
    "gru", "gru_cache", "transformer",
)
BACKBONE_ARMS = frozenset(("lrc_local", "lrc_cache", "gru", "gru_cache", "transformer"))
LEXICAL_ARMS = {
    "v4_lexical": "lexical_address",
    "v4_lexical_slot": "lexical_address_slot",
    "v4_shuffled": "shuffled_lexical_address_slot",
}


def build_model(arm: str) -> nn.Module:
    """Construct an untrained version with approximately 43K allocated weights."""
    if arm not in ARMS:
        raise ValueError(f"unknown benchmark arm: {arm}")
    if arm.startswith("v1_"):
        model = RelationalLM(Config(width=30, layers=2, local_kernel=3,
                                    variant=arm.removeprefix("v1_")))
    elif arm == "v3":
        model = SerialMultiHopLM(Config(width=26, layers=2, local_kernel=3))
    elif arm == "v4_p16":
        model = PagedMultiHopLM(Config(width=22, layers=2, local_kernel=3), page_bits=16)
    elif arm == "v4" or arm in LEXICAL_ARMS:
        model = PagedMultiHopLM(Config(width=24, layers=2, local_kernel=3), page_bits=8)
    else:
        model = BackboneModel(arm)
    model.benchmark_arm = arm
    return model


def train_step(model: nn.Module, optimizer, x, y, seed: int) -> dict:
    """One faithful training update, including its real auxiliary forward work.

    Batches contain real raw-byte IDs 1..256; 0 is reserved for padding.  Source
    lexical-cache preprocessing is shared by both counterfactual forwards for
    the same input and cleared afterwards so another call cannot reuse it.
    """
    arm = model.benchmark_arm
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must have the same [batch, length] shape")
    if (x <= 0).any() or (x >= 257).any() or (y <= 0).any() or (y >= 257).any():
        raise ValueError("the benchmark uses only real raw-byte tokens 1..256")
    if isinstance(model, BackboneModel):
        model.cache_data = cache_batch(x) if model.cached else None
    try:
        if arm.startswith("v1_") or arm == "v3":
            receipt = counterfactual_step(model, optimizer, x, y, seed=seed)
        elif arm in LEXICAL_ARMS:
            receipt = lexical_credit_step(model, optimizer, x, y,
                                          arm=LEXICAL_ARMS[arm], seed=seed)
        elif arm in ("v4", "v4_p16", "lrc_local", "lrc_cache"):
            receipt = hierarchical_credit_step(model, optimizer, x, y, seed=seed)
        elif arm in ("gru", "gru_cache", "transformer"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out.logits.flatten(0, 1), y.flatten())
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient; step cancelled")
            optimizer.step()
            receipt = {"loss": float(loss.detach()), "gradient_norm": float(norm),
                       "forward_passes": 1, "interventions": 0}
        else:
            raise ValueError(f"no backbone fitting recipe for {arm}")
    finally:
        if isinstance(model, BackboneModel):
            model.cache_data = None
    receipt = dict(receipt)
    receipt.setdefault("loss", receipt.get("nll"))
    if receipt["loss"] is None:
        raise RuntimeError("source trainer did not return a next-byte loss")
    receipt["loss"] = float(receipt["loss"])
    return receipt


def build_gated_model(backbone: PagedMultiHopLM, *, variant="utility", gate_state=None):
    """Copy a trained p8 V4 backbone; only the final 50-weight head is new.

    The source utility procedure fixes layer 0 ON.  The caller fits the final
    head on TRAIN-only signed ON/OFF utility and keeps the backbone frozen.
    ``always_on``, ``last_off`` and ``always_off`` are evaluation controls.
    Shuffled/random utility training uses the same inference implementation.
    """
    if not isinstance(backbone, PagedMultiHopLM):
        raise TypeError("utility training reuses an ordinary V4 backbone")
    if variant not in ("utility", "utility_shuffled", "utility_random",
                       "always_on", "last_off", "always_off"):
        raise ValueError(variant)
    model = GatedMultiHopLM(backbone.config, page_bits=backbone.page_bits, mode="learned")
    missing = model.load_state_dict(backbone.state_dict(), strict=False)
    expected = {f"read_gates.{i}.{part}" for i in range(backbone.config.layers)
                for part in ("weight", "bias")}
    if missing.unexpected_keys or set(missing.missing_keys) != expected:
        raise RuntimeError("V4-to-gated copy changed a backbone tensor")
    model.fixed_gates = {0: 1}
    # Match the original deterministic head initialization, including unused
    # first-layer allocation, without depending on an unrelated RNG draw.
    for head in model.read_gates:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        with torch.no_grad():
            head.bias[1] = -0.02
    if gate_state is not None:
        model.read_gates[-1].load_state_dict(gate_state, strict=True)
    if variant == "always_on":
        model.fixed_gates = {i: 1 for i in range(backbone.config.layers)}
    elif variant == "last_off":
        model.fixed_gates = {0: 1, 1: 0}
    elif variant == "always_off":
        model.mode = "always_off"
    model.benchmark_arm = variant
    return model.eval().requires_grad_(False)


def retained_state_bytes(value) -> int:
    """Retained tensor storage and Python containers, excluding model weights.

    This intentionally has the same meaning as the existing backbone meter:
    it is neither allocator peak nor process RSS.  Shared storages count once.
    """
    seen, storages = set(), set()

    def size(obj):
        if obj is None or id(obj) in seen:
            return 0
        seen.add(id(obj))
        n = sys.getsizeof(obj)
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (obj.device.type, obj.device.index, storage.data_ptr())
            if key not in storages:
                storages.add(key)
                n += storage.nbytes()
        elif isinstance(obj, dict):
            n += sum(size(k) + size(v) for k, v in obj.items())
        elif isinstance(obj, (tuple, list, deque)):
            n += sum(size(v) for v in obj)
        elif hasattr(obj, "__dict__"):
            n += size(obj.__dict__)
        return n

    return size(value)


class ByteStream:
    """Uniform batch-one cached-inference adapter, with fresh-state reset."""

    def __init__(self, model):
        if model.training:
            raise ValueError("call model.eval() before make_stream")
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("the historical backbone streaming adapter is CPU-only")
        self.model = model
        self.reset()

    def reset(self):
        m = self.model
        self.position = 0
        if isinstance(m, BackboneModel):
            m.cache_data = None
            self.engine = BackboneStream(m)
            self.already_normalized = True
        elif isinstance(m, GatedMultiHopLM):
            self.engine = LocalOnlyStream(m) if m.mode == "always_off" else StreamingGatedMultiHop(m)
            self.already_normalized = False
        elif isinstance(m, PagedMultiHopLM):
            self.engine = StreamingPagedMultiHop(m)
            self.already_normalized = False
        elif isinstance(m, SerialMultiHopLM):
            self.engine = StreamingMultiHop(m)
            self.already_normalized = False
        elif isinstance(m, RelationalLM):
            self.engine = StreamingInference(m)
            self.already_normalized = False
        else:
            raise TypeError(type(m).__name__)
        return self

    @torch.no_grad()
    def step(self, token: int):
        if isinstance(token, torch.Tensor):
            if token.numel() != 1:
                raise ValueError("ByteStream is batch one")
            token = int(token)
        if not isinstance(token, int) or not 1 <= token <= 256:
            raise ValueError("step expects one real byte token in 1..256")
        z = (self.engine.step(token) if self.already_normalized
             else self.engine.step(torch.tensor([token], dtype=torch.long)))
        self.position += 1
        return z if self.already_normalized else z.log_softmax(-1)

    @torch.no_grad()
    def prefill(self, tokens):
        if self.position:
            raise ValueError("prefill requires a fresh stream; call reset first")
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 1:
                raise ValueError("prefill expects a flat batch-one token sequence")
            tokens = tokens.tolist()
        tokens = list(tokens)
        if not tokens or any(not isinstance(t, int) or not 1 <= t <= 256 for t in tokens):
            raise ValueError("prefill expects a nonempty real-byte sequence")
        z = (self.engine.prefill(tokens) if self.already_normalized else
             self.engine.prefill(torch.tensor([tokens], dtype=torch.long)))
        self.position = len(tokens)
        return z if self.already_normalized else z.log_softmax(-1)

    def state_bytes(self):
        if self.already_normalized:
            return self.engine.state_bytes()
        if isinstance(self.engine, LocalOnlyStream):
            return retained_state_bytes(self.engine.embeds)
        return retained_state_bytes(self.engine.cache)


def make_stream(model):
    return ByteStream(model)
