"""Actual V4 paged circuit with VB-001's sentence encoder; no test labels.

Supervision uses only support occurrence indices, not canonical entity keys.
A training-only all-record address-agreement loss supplies writer gradients.
It is not a prefix-scoring inference path and is explicitly a surrogate.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ..hierarchical_model import PagedMultiHopLM
from ..model import Config
from ..versioned_baselines.study import SentenceEncoder, N_RECORDS


class SentenceLRC(PagedMultiHopLM):
    def __init__(self, width=20):
        super().__init__(Config(vocab=4, width=width, layers=2, relations=4,
                                addresses=8, candidates=2, local_kernel=1), page_bits=3)
        # Replace the byte/local-convolution frontend, not the V4 circuit.
        del self.embedding, self.local, self.local_norm
        self.encoder = SentenceEncoder(width)
        self.position = nn.Embedding(N_RECORDS+1, width)

    def forward(self, records, query, *, sample=False, replay=None, overrides=None,
                noise=None, memory='normal'):
        if records.ndim != 3 or records.shape[1] != N_RECORDS:
            raise ValueError('expected 16 sentence records')
        if query.shape != (records.shape[0], records.shape[-1]):
            raise ValueError('query and records must have matching batches/word counts')
        if memory not in ('normal', 'disabled'):
            raise ValueError('unknown memory mode')
        h = torch.cat((self.encoder(records), self.encoder(query)[:,None]), dim=1)
        h = h + self.position(torch.arange(h.shape[1], device=h.device))
        tokens = torch.ones(h.shape[:2], dtype=torch.long, device=h.device)
        return self._forward_states(tokens, h, tokens.bool(), sample=sample,
                                    replay=replay, overrides=overrides, noise=noise, memory=memory)


def gold_route_overrides(output, supports, candidates=2, page_bits=3):
    """Map support occurrences to CURRENT learned write codes, page and slot.

    Privileged diagnostic only. No canonical semantic address labels are added.
    A second forward with these overrides forces query operands through the
    real capped index; policies still execute and their cost is counted.
    """
    overrides = {}
    batch, n = output.decisions[(0,'write_relation')].shape
    rows = torch.arange(batch, device=supports.device)
    for ell in range(len(output.operands)):
        wr, wa = (output.decisions[(ell,k)] for k in ('write_relation','write_address'))
        routes = output.decisions[(ell,'routes')].clone()
        for hop, name in enumerate(('first','second')):
            target = supports[:,hop]
            rel, addr = wr[rows,target], wa[rows,target]
            newer = torch.arange(n,device=wr.device)[None,:] > target[:,None]
            newer[:,-1] = False  # question occurrence is never an eligible write
            ordinal = (newer & wr.eq(rel[:,None]) & wa.eq(addr[:,None])).sum(-1)
            page, slot = ordinal // candidates, ordinal % candidates
            routes[:,-1,hop] = rel
            for field, value in ((f'query_{name}',addr),(f'rank_{name}',slot)):
                action = output.decisions[(ell,field)].clone()
                action[:,-1] = value
                overrides[(ell,field)] = action
            bits = output.decisions[(ell,f'page_{name}')].clone()
            bits[:,-1] = (page[:,None] >> torch.arange(page_bits,device=page.device)[None,:]) & 1
            overrides[(ell,f'page_{name}')] = bits
        overrides[(ell,'routes')] = routes
    return overrides


def support_loss(output, supports):
    """Identical label information to VB-001; architecture-specific objective.

    Address-agreement contrast normalizes over all 16 records during TRAINING.
    Writer/query soft distributions share latent codes. Page/slot targets are
    derived from support positions and current detached hard write assignments.
    Actual model execution always uses its hard indexed reads.
    """
    rows = torch.arange(supports.shape[0],device=supports.device)
    targets = gold_route_overrides(output,supports)
    losses = []
    for ell in range(len(output.operands)):
        wr = output.policies[(ell,'write_relation')][:,:-1].softmax(-1)
        wa = output.policies[(ell,'write_address')][:,:-1].softmax(-1)
        for hop,name in enumerate(('first','second')):
            qr = output.policies[(ell,'routes')][:,-1,hop].softmax(-1)
            qa = output.policies[(ell,f'query_{name}')][:,-1].softmax(-1)
            score = (wr * qr[:,None]).sum(-1) * (wa * qa[:,None]).sum(-1)
            probs = score / score.sum(-1,keepdim=True).clamp_min(1e-12)
            agreement = -probs[rows,supports[:,hop]].clamp_min(1e-12).log().mean()
            rank = F.cross_entropy(output.policies[(ell,f'rank_{name}')][:,-1],
                                   targets[(ell,f'rank_{name}')][:,-1])
            # Zero-valued high bits are supervised too; inference masks inactive
            # bits. This deterministic coding adds no labels beyond occurrence ID.
            page = F.cross_entropy(output.policies[(ell,f'page_{name}')][:,-1].flatten(0,1),
                                   targets[(ell,f'page_{name}')][:,-1].flatten())
            losses.append(agreement + (rank+page)/2)
    return torch.stack(losses).mean()
