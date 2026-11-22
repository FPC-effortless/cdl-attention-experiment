"""Prospective local role-binding arms; no persistent store or global router."""
from functools import partial
import unittest

import torch
from torch import nn

from nanocog.model import Config, Decoder

from .model import Communication, build_decoder

ARMS = ("gated32", "static32", "attention32", "gated48")

class StaticTaps(nn.Module):
    """Position-dependent weights shared by tokens instead of content gates."""
    def __init__(self, width, window):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(window))

    def forward(self, x):
        return self.logits.expand(*x.shape[:-1], -1)

class LocalAttention(nn.Module):
    """Seven-token batch-only causal competence control, with active rank 10."""
    def __init__(self, width):
        super().__init__()
        self.key = nn.Linear(width, 10, bias=False)
        self.query = nn.Linear(width, 10, bias=False)
        self.value = nn.Linear(width, 10, bias=False)
        self.project = nn.Linear(10, width, bias=False)
        self.relative = nn.Parameter(torch.zeros(7))

    def forward(self, x, state=None):
        if state is not None or not 1 <= x.size(1) <= 7:
            raise ValueError("attention control is for fresh prompts of at most seven tokens")
        idx = torch.arange(x.size(1), device=x.device)
        lag = idx[:, None] - idx[None, :]
        score = (self.query(x) @ self.key(x).transpose(-2, -1)) / (10**0.5)
        score = (score + self.relative[lag.clamp_min(0)]).masked_fill(lag < 0, -torch.inf)
        return self.project(score.softmax(dim=-1) @ self.value(x)), None

def new_mixer(width, arm):
    mod = Communication(width, "A")
    if arm == "static32":
        mod.local.offset_gate = StaticTaps(width, mod.local.window)
    elif arm == "attention32":
        mod.local = LocalAttention(width)
    return mod

def build_arm(vocab, arm, seed):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    width = 48 if arm == "gated48" else 32
    ref = build_decoder(vocab, "A", width=width, layers=2, seed=seed)
    if arm.startswith("gated"):
        return ref
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = Decoder(Config(vocab, width=width, layers=2, vocab_alignment=8),
                    partial(new_mixer, arm=arm))
    src, dest = ref.state_dict(), m.state_dict()
    # Exactly identical initial trunk for all width-32 comparisons. Untouched
    # convolution tensors stay shared in the static-gate intervention as well.
    for name, old in src.items():
        if name in dest and old.shape == dest[name].shape:
            dest[name] = old
    if arm == "static32":
        # Compare to native initialization with its content term set to zero.
        # Transfer the offset bias; remove, rather than pad, the dynamic weights.
        for i in range(2):
            dest[f"mixers.{i}.local.offset_gate.logits"] = src[f"mixers.{i}.local.offset_gate.bias"]
    m.load_state_dict(dest, strict=True)
    return m

class MixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_active_parameter_budget(self):
        counts = {a: sum(p.numel() for p in build_arm(287,a,1171).parameters()) for a in ARMS}
        self.assertEqual(counts, dict(gated32=37414, static32=37158, attention32=37420, gated48=69926))

    def test_strict_batch_causality(self):
        x = torch.tensor([[262,263,264,265,266,267,268]], dtype=torch.long)
        for arm in ARMS:
            m = build_arm(287,arm,1171).eval()
            with torch.inference_mode():
                base,_ = m(x)
                for t in range(1,7):
                    y=x.clone()
                    y[:,t:]=269
                    edited,_=m(y)
                    torch.testing.assert_close(base[:,:t],edited[:,:t],rtol=1e-5,atol=1e-6)
                    short,_=m(x[:,:t])
                    torch.testing.assert_close(base[:,:t],short,rtol=1e-5,atol=1e-6)

    def test_one_change_trunk(self):
        ref = build_arm(287,"gated32",1171).state_dict()
        for arm in ("static32","attention32"):
            other = build_arm(287,arm,1171).state_dict()
            for k,t in ref.items():
                if not k.startswith("mixers.") or (arm=="static32" and ".offset_gate." not in k):
                    self.assertTrue(torch.equal(t,other[k]),k)

    def test_static_equivalent_to_zero_conditioning(self):
        x = torch.tensor([[262,263,264,265,266,267,268]], dtype=torch.long)
        ref = build_arm(287,"gated32",1171).eval()
        static = build_arm(287,"static32",1171).eval()
        with torch.inference_mode():
            for b in ref.mixers:
                b.local.offset_gate.weight.zero_()
            torch.testing.assert_close(ref(x)[0],static(x)[0],rtol=1e-6,atol=1e-7)

if __name__ == "__main__":
    unittest.main()
