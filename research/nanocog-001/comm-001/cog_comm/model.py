"""Nested COG-COMM arms in the checked nanoCOG decoder shell."""
from functools import partial
import torch
from torch import nn
from nanocog.model import Config, Decoder
from .mix import LocalMix, SummaryMix
from .router import BoundedRoute


class Communication(nn.Module):
    def __init__(self, width, arm, block=8, route_enabled=True, route_mode="learned", retained_blocks=16):
        super().__init__()
        if arm not in ("A", "B", "C"):
            raise ValueError("arm must be A, B or C")
        self.arm = arm
        self.route_enabled = route_enabled
        self.local = LocalMix(width)
        if arm in ("B", "C"):
            self.summary = SummaryMix(width, block=block)
        if arm == "C":
            self.route = BoundedRoute(width, block=block, route_mode=route_mode, retained_blocks=retained_blocks)

    def forward(self, x, state=None):
        lc, sc, rc = (None, None, None) if state is None else state
        out, lc = self.local(x, lc)
        if self.arm in ("B", "C"):
            y, sc = self.summary(x, sc)
            out = out + y
        if self.arm == "C":
            y, rc = self.route(x, rc, disable=not self.route_enabled)
            out = out + y
        return out, (lc, sc, rc)


def build_decoder(vocab_size, arm, *, width=32, layers=2, block=8, seed=171, route_enabled=True, route_mode="learned", retained_blocks=16):
    """Match initial shared tensors despite any additional per-arm layers."""
    cfg = Config(vocab_size, width=width, layers=layers, vocab_alignment=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = Decoder(cfg, partial(Communication, arm="A", block=block))
        if arm == "A":
            return base
        torch.manual_seed(seed)
        candidate = Decoder(cfg, partial(Communication, arm=arm, block=block, route_enabled=route_enabled, route_mode=route_mode, retained_blocks=retained_blocks))
        init = candidate.state_dict()
        for name, value in base.state_dict().items():
            if name not in init or init[name].shape != value.shape:
                raise ValueError("base arm geometry no longer matches")
            init[name] = value
        if arm == "C":
            torch.manual_seed(seed)
            b = Decoder(cfg, partial(Communication, arm="B", block=block))
            for name, value in b.state_dict().items():
                if ".summary." in name:
                    init[name] = value
        candidate.load_state_dict(init,strict=True)
    return candidate


def set_routing(model, enabled):
    """Same-checkpoint route effect. Call only across separate full resets."""
    for m in model.mixers:
        m.route_enabled = bool(enabled)
