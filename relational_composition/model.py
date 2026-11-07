"""Small raw-byte LM with hard indexed reads and ordered composition.

All decisions have fixed-size categorical spaces, independent of context length.
No straight-through routing gradient, prefix similarity matrix, entity parser,
gold address, or supplied graph enters forward(). See REVIEW.md for limitations.
"""
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .index import CausalIndex


@dataclass(frozen=True)
class Config:
    vocab: int = 257  # UTF-8 bytes plus padding; no learned tokenizer/UNK collapse
    width: int = 32
    layers: int = 2
    relations: int = 4
    addresses: int = 8
    candidates: int = 2
    local_kernel: int = 5
    variant: Literal["coupled", "independent", "no_reference"] = "coupled"

    def __post_init__(self):
        if min(self.vocab, self.width, self.layers, self.relations,
               self.addresses, self.candidates, self.local_kernel) < 1:
            raise ValueError("all dimensions must be positive")
        if self.variant not in ("coupled", "independent", "no_reference"):
            raise ValueError("unknown variant")


@dataclass
class Output:
    logits: torch.Tensor
    decisions: dict
    policies: dict
    noise: dict
    references: list[torch.Tensor]
    operands: list[torch.Tensor]
    metrics: dict[str, int]


def dereference(h, references):
    """-1 is explicit null, never Python's last-element indexing convention."""
    rows = torch.arange(h.shape[0], device=h.device)[:, None]
    values = h[rows, references.clamp_min(0)]
    return values * (references >= 0).unsqueeze(-1)


class Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d, r, a = c.width, c.relations, c.addresses
        self.write_relation = nn.Linear(d, r)
        self.write_address = nn.Linear(d, a)
        self.route = nn.Linear(2 * d, 2 * r)
        self.relation = nn.Embedding(r, d)
        self.query = nn.Linear(3 * d, 2 * a)
        self.rank = nn.Linear(3 * d, 2 * (c.candidates + 1))
        self.continue_ref = nn.Linear(4 * d, 4)  # null, previous, operand 1, operand 2
        self.compose = nn.Sequential(nn.Linear(4 * d + 2, 2 * d), nn.GELU(),
                                     nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)


class RelationalLM(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.width, padding_idx=0)
        self.local = nn.Conv1d(config.width, config.width, config.local_kernel)
        self.local_norm = nn.LayerNorm(config.width)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.layers))
        self.lm_head = nn.Linear(config.width, config.vocab)

    def forward(self, tokens, *, sample=False, replay=None, overrides=None, noise=None,
                memory: Literal["normal", "disabled", "shuffled"] = "normal"):
        """Each row is ONE document, right padded. No cross-document state.

        Replay freezes discrete actions, not retrieved identities: interventions
        on writes still rebuild postings and change downstream operands. Overrides
        are for training interventions/integrity tests, never evaluation routing.
        Shuffled control corrupts retrieved identity by a causal prefix rotation;
        it uses the same number of feature gathers and never reads future values.
        """
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("expected a nonempty [document, time] token tensor")
        if memory not in ("normal", "disabled", "shuffled"):
            raise ValueError("unknown memory control")
        valid = tokens != 0
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("padding must be a suffix; use one document per row")
        c = self.config
        b, n = tokens.shape
        decisions, policies, reference_trace, operand_trace = {}, {}, [], []
        used_noise = {}
        metrics = dict(writes=0, reads=0, candidates=0, omitted=0, null_reads=0,
                       reference_dereferences=0, operator_executions=0)
        overrides = overrides or {}

        def choose(layer, name, logits):
            key = (layer, name)
            policies[key] = logits
            if key in overrides:
                value = overrides[key].to(tokens.device)
            elif replay is not None and key in replay:
                value = replay[key].to(tokens.device)
            elif sample:
                perturbation = None if noise is None else noise.get(key)
                if perturbation is None:
                    uniform = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
                    perturbation = -(-uniform.log()).log()
                if perturbation.shape != logits.shape:
                    raise ValueError("routing noise shape mismatch")
                used_noise[key] = perturbation
                value = (logits + perturbation).argmax(-1)
            else:
                value = logits.argmax(-1)
            if value.shape != logits.shape[:-1] or (value < 0).any() or (value >= logits.shape[-1]).any():
                raise ValueError(f"invalid discrete action: {key}")
            decisions[key] = value.detach().clone()
            return value

        h = self.embedding(tokens)
        h = self.local_norm(h + F.gelu(self.local(
            F.pad(h.transpose(1, 2), (c.local_kernel - 1, 0))).transpose(1, 2)))
        h = h * valid.unsqueeze(-1)
        refs = torch.full_like(tokens, -1)
        for ell, layer in enumerate(self.layers):
            # h is immutable for this complete layer. No same-layer token scan.
            anchor = dereference(h, refs)
            metrics["reference_dereferences"] += int((refs >= 0).sum())
            wr = choose(ell, "write_relation", layer.write_relation(h))
            wa = choose(ell, "write_address", layer.write_address(h))
            route_logits = layer.route(torch.cat((h, anchor), -1)).reshape(b, n, 2, c.relations)
            if c.variant == "independent":
                rr = choose(ell, "read_relation", route_logits[:, :, 0])
                tr = choose(ell, "transform_relation", route_logits[:, :, 1])
            else:
                # Both heads participate; all variants have identical parameters.
                rr = tr = choose(ell, "relation", route_logits.mean(-2))
            route_state = torch.cat((h, anchor, layer.relation(rr)), -1)
            query = choose(ell, "query", layer.query(route_state).reshape(b, n, 2, c.addresses))
            ranks = choose(ell, "rank", layer.rank(route_state).reshape(b, n, 2, c.candidates + 1))
            operand_ids = torch.full((b, n, 2), -1, dtype=torch.long, device=tokens.device)
            # Explicit CPU index costs/synchronization; not advertised as fast GPU code.
            wrl, wal, rrl, ql, kl, vl = [v.detach().cpu().tolist() for v in (wr, wa, rr, query, ranks, valid)]
            for row in range(b):
                index = CausalIndex(c.candidates)
                for t in range(n):
                    if vl[row][t]:
                        index.append(t, wrl[row][t], wal[row][t])
                        metrics["writes"] += 1
                for t in range(n):
                    if not vl[row][t]:
                        continue
                    for k in range(2):
                        read = index.lookup(rrl[row][t], ql[row][t][k], t)
                        metrics["reads"] += 1
                        metrics["candidates"] += len(read.candidates)
                        metrics["omitted"] += read.omitted
                        rank = kl[row][t][k]
                        if memory != "disabled" and rank < len(read.candidates):
                            identity = read.candidates[rank]
                            if memory == "shuffled":
                                identity = (identity + max(1, t // 2)) % t
                            operand_ids[row, t, k] = identity
                        else:
                            metrics["null_reads"] += 1
            left = dereference(h, operand_ids[:, :, 0])
            right = dereference(h, operand_ids[:, :, 1])
            transform = layer.relation(tr)
            continuation = choose(ell, "continuation", layer.continue_ref(
                torch.cat((h, left, right, transform), -1)))
            options = torch.stack((torch.full_like(refs, -1), refs,
                                   operand_ids[:, :, 0], operand_ids[:, :, 1]), -1)
            refs = options.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
            if c.variant == "no_reference" or memory == "disabled":
                refs = torch.full_like(refs, -1)
            refs = torch.where(valid, refs, -1)
            h = layer.norm(h + layer.compose(torch.cat((h, left, right, transform,
                (operand_ids >= 0).to(h.dtype)), -1))) * valid.unsqueeze(-1)
            metrics["operator_executions"] += int(valid.sum())
            reference_trace.append(refs)
            operand_trace.append(operand_ids)
        return Output(self.lm_head(h), decisions, policies, used_noise, reference_trace, operand_trace, metrics)


def language_loss(output: Output, targets: torch.Tensor):
    if not (targets != 0).any():
        raise ValueError("at least one non-padding target is required")
    return F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), ignore_index=0)
