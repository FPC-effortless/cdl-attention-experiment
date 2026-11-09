"""Regulated CSM: a NumPy hard-forward mathematical reference.

This is a new indexed, float64 reference of the published mathematical equations,
not a port certified against the archived PyTorch implementation. No training,
surrogate backward rule, performance claim, or model-quality result is included.
The original write/readout/cursor/link semantics are retained. The only candidate
architectural addition is a token-only regulatory trace bank in the controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Config:
    slots: int = 8
    width: int = 32
    operations: int = 4
    hidden: int = 64
    vocabulary: int = 19
    traces: int = 12


@dataclass(frozen=True)
class State:
    values: np.ndarray
    links: np.ndarray
    cursor: int
    older: np.ndarray
    recent: np.ndarray
    position: int
    regulation: np.ndarray


def initialize(c: Config) -> State:
    if min(c.slots, c.width, c.operations, c.hidden, c.vocabulary) <= 0 or c.traces < 0:
        raise ValueError("Invalid model dimensions")
    return State(np.zeros((c.slots, c.width)), np.arange(c.slots), 0,
                 np.zeros(c.width), np.zeros(c.width), 0, np.zeros(c.traces))


def parameters(c: Config, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p: Dict[str, np.ndarray] = {}

    def linear(name: str, nin: int, nout: int) -> None:
        p[name + ".W"] = rng.normal(0, 1 / math.sqrt(nin), (nout, nin))
        p[name + ".b"] = rng.normal(0, 0.08, nout)

    d, s, o, h, v = c.width, c.slots, c.operations, c.hidden, c.vocabulary
    p["embedding"] = rng.normal(0, 0.3, (v, d))
    linear("frontend", 3 * d, d)
    linear("controller", 2 * d, d)
    linear("first", d, s)
    linear("second", 2 * d, s)
    linear("destination", d, s)
    linear("operator", d, o)
    linear("use_link", d, 2)
    linear("value_edit", d, 2)
    linear("link_edit", d, 2)
    linear("gain", d, 1)
    p["operation_embedding"] = rng.normal(0, 0.3, (o, d))
    linear("transform1", 4 * d, h)
    linear("transform2", h, d)
    linear("output", d, v)
    # Appending these parameters keeps the seeded base parameters identical.
    if c.traces:
        linear("trace_driver", d, c.traces)
        p["trace_to_controller"] = rng.normal(0, 1 / math.sqrt(c.traces), (d, c.traces))
    return p


def linear(p: Dict[str, np.ndarray], name: str, x: np.ndarray) -> np.ndarray:
    return p[name + ".W"] @ x + p[name + ".b"]


def decay(c: Config, variant: str) -> np.ndarray:
    if not c.traces:
        return np.empty(0)
    if variant == "instant":
        return np.zeros(c.traces)
    if variant == "single":
        return np.full(c.traces, math.exp(-1 / 32))
    if variant == "multi":
        if c.traces % 3:
            raise ValueError("Three equally sized groups require traces divisible by 3")
        return np.repeat(np.exp(-1 / np.array([4., 32., 256.])), c.traces // 3)
    raise ValueError("Unknown trace variant")


def _select(logits: np.ndarray) -> int:
    # NumPy argmax breaks ties by lowest index. Softmax at any positive
    # temperature has the same argmax for finite exact-real logits.
    return int(np.argmax(logits))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def step(c: Config, p: Dict[str, np.ndarray], state: State, token: int,
         lam: np.ndarray, mode: str = "rewrite", valid: bool = True,
         reset: bool = False) -> Tuple[State, np.ndarray | None, dict]:
    """One observed-token transition; padding is identity, reset starts a document.

    reset happens before a valid token. reset+padding is rejected to avoid
    an ambiguous boundary convention. Inputs and parameters must be finite;
    float64 tests do not establish arbitrary IEEE-runtime refinement.
    """
    if reset and not valid:
        raise ValueError("A reset must coincide with an observed token")
    if not valid:
        return state, None, {"padding": True}
    if reset:
        state = initialize(c)
    if mode not in {"rewrite", "ring_write", "no_links"}:
        raise ValueError("Unknown mode")
    if not (0 <= token < c.vocabulary) or lam.shape != (c.traces,):
        raise ValueError("Invalid token or decay shape")
    if np.any(lam < 0) or np.any(lam > 1) or not np.isfinite(lam).all():
        raise ValueError("Decay must lie in [0,1]")
    if not (0 <= state.cursor < c.slots):
        raise ValueError("Invalid cursor")
    if np.any(state.links < 0) or np.any(state.links >= c.slots):
        raise ValueError("Invalid link")
    arrays = [state.values, state.older, state.recent, state.regulation, *p.values()]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite inputs are outside this reference contract")

    e = p["embedding"][token]
    front = np.tanh(linear(p, "frontend", np.concatenate((state.older, state.recent, e))))
    if c.traces:
        signal = np.tanh(linear(p, "trace_driver", front))
        r = lam * state.regulation + (1 - lam) * signal
        injection = p["trace_to_controller"] @ r
    else:
        r, injection = np.empty(0), np.zeros(c.width)
    u = np.tanh(linear(p, "controller", np.concatenate((front, state.values[state.cursor]))) + injection)
    i = _select(linear(p, "first", u))
    a = state.values[i]
    jdirect = _select(linear(p, "second", np.concatenate((u, a))))
    uselink = _select(linear(p, "use_link", u))
    j = jdirect if mode == "no_links" or not uselink else int(state.links[i])
    b = state.values[j]
    k = _select(linear(p, "operator", u))
    op = p["operation_embedding"][k]
    proposal = np.tanh(linear(p, "transform2", np.tanh(
        linear(p, "transform1", np.concatenate((front, a, b, op))))))
    dest = state.position % c.slots if mode == "ring_write" else _select(linear(p, "destination", u))
    admitted = bool(_select(linear(p, "value_edit", u)))
    gain = _sigmoid(float(linear(p, "gain", u)[0]))
    linkedit = bool(_select(linear(p, "link_edit", u)))
    old = state.values[dest]
    written = old + gain * (proposal - old) if admitted else old.copy()
    values, links = state.values.copy(), state.links.copy()
    values[dest] = written
    # Preserve original no_links semantics: traversal disabled, edits still run.
    if linkedit:
        links[dest] = j
    nxt = State(values, links, dest, state.recent.copy(), e.copy(), state.position + 1, r)
    logits = linear(p, "output", written)
    trace = {"first": i, "second": j, "destination": dest, "operator": k,
             "admitted": admitted, "gain": gain, "link_edit": linkedit,
             "proposal": proposal.copy(), "written": written.copy()}
    return nxt, logits, trace


def run(c: Config, p: Dict[str, np.ndarray], tokens: List[int], lam: np.ndarray,
        mode: str = "rewrite", state: State | None = None):
    st = initialize(c) if state is None else state
    outputs = []
    for token in tokens:
        st, logits, _ = step(c, p, st, token, lam, mode)
        outputs.append(logits)
    return st, np.stack(outputs) if outputs else np.empty((0, c.vocabulary))


def counts(c: Config) -> dict:
    s, d, o, h, v, m = c.slots, c.width, c.operations, c.hidden, c.vocabulary, c.traces
    base_p = 2*v*d+v+5*d*d+4*s*d+3*s+2*o*d+o+5*d*h+h+10*d+7
    base_mac = 5*d*d+8*s*d+2*o*d+5*d*h+d*v+s*s+7*d
    base_cache = s*d+s*s+s+2*d
    return {"base_parameters": base_p, "added_parameters": 2*d*m+m,
            "candidate_parameters": base_p+2*d*m+m,
            "base_dense_nominal_macs": base_mac,
            "added_dense_nominal_macs": 2*d*m,
            "candidate_dense_nominal_macs": base_mac+2*d*m,
            "base_dense_cache_fp32_bytes": 4*base_cache,
            "added_trace_fp32_bytes": 4*m,
            "candidate_dense_cache_fp32_bytes": 4*(base_cache+m),
            "note": "Dense-source accounting; reference uses indexed links and float64. Excludes elementwise arithmetic, memory traffic, backward, allocator, and position integer."}
