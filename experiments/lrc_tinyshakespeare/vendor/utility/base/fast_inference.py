"""Exact-in-the-limit cached autoregressive inference for LRC-001 RelationalLM.

Drop into the existing ``relational_composition`` package alongside model.py/index.py.
This adapter preserves the v1 learned policy and weight tensor shapes: it is a
systems change, not a demonstrated improvement in reasoning. It uses strict
causality to compute each newly appended token without recomputing old tokens.

The reference kernel keeps CPU posting lists and calls small PyTorch operators;
its asymptotics improve, but latency/throughput must be measured on target HW.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import torch
from torch.nn import functional as F

from .index import CausalIndex
from .model import RelationalLM


@dataclass
class GrowableStates:
    """Amortized O(1) append and O(1) position lookup (times feature width)."""
    values: torch.Tensor
    length: int = 0

    @classmethod
    def empty(cls, batch_size: int, width: int, device, dtype):
        return cls(torch.empty(batch_size, 16, width, device=device, dtype=dtype))

    def load(self, full: torch.Tensor):
        """Fill from a vectorized prefill's [B, N, d] hidden states."""
        b, n, d = full.shape
        capacity = max(16, 1 << (n-1).bit_length())
        self.values = full.new_empty((b, capacity, d))
        self.values[:, :n].copy_(full)
        self.length = n

    def append(self, value: torch.Tensor):
        if self.length == self.values.shape[1]:
            larger = self.values.new_empty((self.values.shape[0], 2 * self.values.shape[1],
                                            self.values.shape[2]))
            larger[:, :self.length].copy_(self.values[:, :self.length])
            self.values = larger
        self.values[:, self.length].copy_(value)
        self.length += 1


@dataclass
class Cache:
    """Prefix state, with one posting index per row and layer.

    states[ell][t] = h^ell_t (the previous-layer input at layer ell).
    Index keys are writer decisions computed from exactly these h^ell_t.
    references are occurrence positions, never hidden-state object identities.
    """
    batch_size: int
    layers: int
    kernel: int
    cap: int
    position: int = 0
    token_embeds: deque[torch.Tensor] = field(default_factory=deque)
    states: list[GrowableStates] = field(default_factory=list)
    postings: list[list[CausalIndex]] = field(default_factory=list)
    count: dict[str, int] = field(default_factory=lambda: dict(writes=0, reads=0,
                                      candidates=0, omitted=0, null_reads=0,
                                      reference_dereferences=0, operator_executions=0))

    @classmethod
    def empty(cls, config, batch_size: int, device, dtype):
        return cls(batch_size=batch_size, layers=config.layers,
                   kernel=config.local_kernel, cap=config.candidates,
                   token_embeds=deque(maxlen=max(config.local_kernel - 1, 0)),
                   states=[GrowableStates.empty(batch_size, config.width, device, dtype)
                           for _ in range(config.layers)],
                   postings=[[CausalIndex(config.candidates) for _ in range(batch_size)]
                             for _ in range(config.layers)],
                   )


class StreamingInference:
    """Greedy, no-grad, unpadded equal-length-document decode engine.

    ``step`` consumes one non-PAD byte token for each document. It returns next
    byte logits, exactly as ``model(prefix).logits[:, -1]`` up to FP arithmetic.
    Work per token does not depend on the *length* of the preceding prefix other
    than logarithmic posting-list lookup and retained O(L*N*d) state storage.
    """

    def __init__(self, model):
        if model.training:
            raise ValueError("call model.eval() before initializing streaming inference")
        self.model = model
        self.config = model.config
        self.cache: Cache | None = None

    def reset(self, batch_size: int, device=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = next(self.model.parameters()).device
        self.cache = Cache.empty(self.config, batch_size, device, self.model.embedding.weight.dtype)
        if isinstance(self.model, RelationalLM):
            # LRC-002's normal path exposes an explicit zero oracle-read counter.
            # SerialMultiHopLM/PagedMultiHopLM have separate metric schemas.
            self.cache.count["oracle_operand_reads"] = 0
        return self

    @staticmethod
    def _deref(states: GrowableStates, refs, template):
        result = template.new_zeros(template.shape)
        if states.length == 0:
            return result
        rows = torch.arange(template.shape[0], device=template.device)
        actual = states.values[rows, refs.clamp_min(0)]
        return torch.where((refs >= 0)[:, None], actual, result)

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 1:
            raise ValueError("step expects one [batch] tensor of token ids")
        if (tokens <= 0).any() or (tokens >= self.config.vocab).any():
            raise ValueError("step requires non-PAD, in-vocabulary tokens")
        if self.cache is None:
            self.reset(int(tokens.numel()), tokens.device)
        cache = self.cache
        if tokens.shape[0] != cache.batch_size:
            raise ValueError("batch size changed without reset")
        model, c = self.model, self.config
        t, batch = cache.position, cache.batch_size
        token_embedding = model.embedding(tokens)
        k = c.local_kernel
        window = list(cache.token_embeds)
        if len(window) + 1 < k:
            padding = [token_embedding.new_zeros(token_embedding.shape)
                       for _ in range(k - 1 - len(window))]
            window = padding + window
        window_tensor = torch.stack(window + [token_embedding], dim=-1)
        conv = F.conv1d(window_tensor, model.local.weight, model.local.bias).squeeze(-1)
        h = model.local_norm(token_embedding + F.gelu(conv))
        cache.token_embeds.append(token_embedding)
        # References carry ACROSS LAYERS at the same position, not across time.
        refs = torch.full((batch,), -1, device=h.device, dtype=torch.long)
        for ell, layer in enumerate(model.layers):
            prior_states = cache.states[ell]
            anchor = self._deref(prior_states, refs, h)
            cache.count["reference_dereferences"] += int((refs >= 0).sum())
            wr = layer.write_relation(h).argmax(-1)
            wa = layer.write_address(h).argmax(-1)
            relation_logits = layer.route(torch.cat((h, anchor), -1)).reshape(
                batch, 2, c.relations)
            if c.variant == "independent":
                rr = relation_logits[:, 0].argmax(-1)
                tr = relation_logits[:, 1].argmax(-1)
            else:
                rr = tr = relation_logits.mean(1).argmax(-1)
            route_state = torch.cat((h, anchor, layer.relation(rr)), -1)
            addresses = layer.query(route_state).reshape(batch, 2, c.addresses).argmax(-1)
            ranks = layer.rank(route_state).reshape(batch, 2, c.candidates + 1).argmax(-1)
            # The writer is inserted for this time before queries, but predecessor
            # filtering with ``before=t`` ensures self/future cannot be read.
            wr_cpu, wa_cpu, rr_cpu = wr.tolist(), wa.tolist(), rr.tolist()
            address_cpu, rank_cpu = addresses.tolist(), ranks.tolist()
            operands = torch.full((batch, 2), -1, device=h.device, dtype=torch.long)
            for b in range(batch):
                ix = cache.postings[ell][b]
                ix.append(t, wr_cpu[b], wa_cpu[b])
                cache.count["writes"] += 1
                for head in range(2):
                    read = ix.lookup(rr_cpu[b], address_cpu[b][head], t)
                    cache.count["reads"] += 1
                    cache.count["candidates"] += len(read.candidates)
                    cache.count["omitted"] += read.omitted
                    rank = rank_cpu[b][head]
                    if rank < len(read.candidates):
                        operands[b, head] = read.candidates[rank]
                    else:
                        cache.count["null_reads"] += 1
            left = self._deref(prior_states, operands[:, 0], h)
            right = self._deref(prior_states, operands[:, 1], h)
            transform = layer.relation(tr)
            continuation = layer.continue_ref(
                torch.cat((h, left, right, transform), -1)).argmax(-1)
            options = torch.stack((torch.full_like(refs, -1), refs,
                                   operands[:, 0], operands[:, 1]), -1)
            new_refs = options.gather(-1, continuation[:, None]).squeeze(-1)
            if c.variant == "no_reference":
                new_refs = torch.full_like(refs, -1)
            next_h = layer.norm(h + layer.compose(torch.cat(
                (h, left, right, transform, (operands >= 0).to(h.dtype)), -1)))
            cache.count["operator_executions"] += batch
            prior_states.append(h)
            refs = new_refs
            h = next_h
        cache.position += 1
        return model.lm_head(h)

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Vectorized full-prompt prefill plus persistent index/cache construction.

        Captures per-layer hidden inputs and writer decisions from the *original*
        causal bulk forward, so prefill is not penalized by token-at-a-time ops.
        Hooks are temporary and make this method non-reentrant across callers
        sharing a model. Only normal greedy evaluation is supported.
        """
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("prefill needs nonempty [batch, length] prompt")
        if (tokens <= 0).any() or (tokens >= self.config.vocab).any():
            raise ValueError("prefill expects unpadded in-vocabulary tokens")
        self.reset(tokens.shape[0], tokens.device)
        b, n = tokens.shape
        captured_h = [None] * self.config.layers
        captured_wr = [None] * self.config.layers
        captured_wa = [None] * self.config.layers
        hooks = []

        def on_writer(ell, name):
            def hook(_module, args, output):
                if name == "relation":
                    captured_h[ell] = args[0]
                    captured_wr[ell] = output.argmax(-1)
                else:
                    captured_wa[ell] = output.argmax(-1)
            return hook

        try:
            for ell, layer in enumerate(self.model.layers):
                hooks.append(layer.write_relation.register_forward_hook(
                    on_writer(ell, "relation")))
                hooks.append(layer.write_address.register_forward_hook(
                    on_writer(ell, "address")))
            out = self.model(tokens)
        finally:
            for handle in hooks:
                handle.remove()

        cache = self.cache
        for ell in range(self.config.layers):
            if any(v is None for v in (captured_h[ell], captured_wr[ell], captured_wa[ell])):
                raise RuntimeError("model did not emit one writer decision per layer")
            cache.states[ell].load(captured_h[ell])
            relations = captured_wr[ell].tolist()
            addresses = captured_wa[ell].tolist()
            for row in range(b):
                index = cache.postings[ell][row]
                for t in range(n):
                    index.append(t, relations[row][t], addresses[row][t])
        k = self.config.local_kernel
        if k > 1:
            suffix_embeds = self.model.embedding(tokens[:, -min(k-1,n):])
            for t in range(suffix_embeds.shape[1]):
                cache.token_embeds.append(suffix_embeds[:, t])
        cache.position = n
        cache.count = dict(out.metrics)
        return out.logits[:, -1]

    @torch.no_grad()
    def prefill_incremental_reference(self, tokens: torch.Tensor) -> torch.Tensor:
        """Slow O(N) stream-prompt reference useful for differential testing."""
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("prefill needs nonempty [batch, length] prompt")
        self.reset(tokens.shape[0], tokens.device)
        last = None
        for t in range(tokens.shape[1]):
            last = self.step(tokens[:, t])
        return last

    @torch.no_grad()
    def greedy_generate(self, tokens: torch.Tensor, new_tokens: int):
        """Reference generation for a fixed-size batch of equal-length prompts."""
        if new_tokens < 0:
            raise ValueError("new_tokens must be nonnegative")
        logits = self.prefill(tokens)
        continuation = []
        for t in range(new_tokens):
            nxt = logits.argmax(-1)
            continuation.append(nxt)
            if t + 1 < new_tokens:
                logits = self.step(nxt)
        return (torch.stack(continuation, 1) if continuation else
                tokens.new_empty((tokens.shape[0], 0)))

