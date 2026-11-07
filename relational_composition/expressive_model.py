"""Research-only LRC-V3: independent relation roles and genuinely dependent reads.

The two retrieval operands may use DIFFERENT relation namespaces. Query 2 is
conditioned on the representation actually returned by query 1. This gives a
structural 2-hop capacity within each layer; it does NOT prove that the model
has learned entity linking or compositional reasoning from natural language.

All categorical policies remain trainable by the existing one-coordinate
counterfactual surrogate (``train.py`` from LRC-001). No oracle features enter
``forward``. The reference bulk implementation retains Python posting loops.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .index import CausalIndex
from .model import Config, Output, dereference
from .fast_inference import StreamingInference


class ProgramLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, r, a, c = cfg.width, cfg.relations, cfg.addresses, cfg.candidates
        self.write_relation = nn.Linear(d, r)
        self.write_address = nn.Linear(d, a)
        self.route = nn.Linear(2*d, 3*r)  # r_first, r_second, transform
        self.relation = nn.Embedding(r, d)
        self.query_first = nn.Linear(3*d, a)
        self.rank_first = nn.Linear(3*d, c+1)
        self.query_second = nn.Linear(4*d, a)
        self.rank_second = nn.Linear(4*d, c+1)
        self.continue_ref = nn.Linear(4*d, 4)
        self.compose = nn.Sequential(nn.Linear(6*d+2, 2*d), nn.GELU(),
                                     nn.Linear(2*d, d))
        self.norm = nn.LayerNorm(d)


class SerialMultiHopLM(nn.Module):
    """Two dependent, separately typed hops per layer, no attention over prefix."""
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.width, padding_idx=0)
        self.local = nn.Conv1d(config.width, config.width, config.local_kernel)
        self.local_norm = nn.LayerNorm(config.width)
        self.layers = nn.ModuleList(ProgramLayer(config) for _ in range(config.layers))
        self.lm_head = nn.Linear(config.width, config.vocab)

    def forward(self, tokens, *, sample=False, replay=None, overrides=None, noise=None,
                memory='normal'):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('expected nonempty [batch, time] tokens')
        if memory not in ('normal', 'disabled'):
            raise ValueError('unsupported memory mode')
        valid = tokens != 0
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError('padding must be a suffix')
        batch, n = tokens.shape
        c = self.config
        overrides = overrides or {}
        policies, decisions, used_noise = {}, {}, {}
        metrics = dict(writes=0, reads=0, candidates=0, omitted=0,
                       null_reads=0, reference_dereferences=0, operator_executions=0)
        ref_history, operand_history = [], []

        def choose(ell, name, logits):
            key = ell, name
            policies[key] = logits
            if key in overrides:
                result = overrides[key].to(tokens.device)
            elif replay is not None and key in replay:
                result = replay[key].to(tokens.device)
            elif sample:
                g = noise.get(key) if noise is not None else None
                if g is None:
                    u = torch.rand_like(logits).clamp_(1e-7, 1-1e-7)
                    g = -(-u.log()).log()
                if g.shape != logits.shape:
                    raise ValueError('routing Gumbel noise shape mismatch')
                used_noise[key] = g
                result = (logits + g).argmax(-1)
            else:
                result = logits.argmax(-1)
            if (result.shape != logits.shape[:-1] or (result < 0).any()
                    or (result >= logits.shape[-1]).any()):
                raise ValueError(f'invalid categorical action {key}')
            decisions[key] = result.detach().clone()
            return result

        def read_postings(indexes, relations, addresses, ranks):
            operands = torch.full((batch, n), -1, device=tokens.device, dtype=torch.long)
            rel_cpu, adr_cpu, rank_cpu = [v.detach().cpu().tolist()
                                          for v in (relations, addresses, ranks)]
            for row in range(batch):
                idx = indexes[row]
                for t in range(n):
                    if not bool(valid[row, t]):
                        continue
                    hit = idx.lookup(rel_cpu[row][t], adr_cpu[row][t], t)
                    metrics['reads'] += 1
                    metrics['candidates'] += len(hit.candidates)
                    metrics['omitted'] += hit.omitted
                    k = rank_cpu[row][t]
                    if memory != 'disabled' and k < len(hit.candidates):
                        operands[row, t] = hit.candidates[k]
                    else:
                        metrics['null_reads'] += 1
            return operands

        h = self.embedding(tokens)
        h = self.local_norm(h + F.gelu(self.local(F.pad(
            h.transpose(1, 2), (c.local_kernel-1, 0))).transpose(1, 2)))
        h = h * valid.unsqueeze(-1)
        refs = torch.full_like(tokens, -1)
        for ell, layer in enumerate(self.layers):
            anchor = dereference(h, refs)
            metrics['reference_dereferences'] += int((refs >= 0).sum())
            wr = choose(ell, 'write_relation', layer.write_relation(h))
            wa = choose(ell, 'write_address', layer.write_address(h))
            routes = choose(ell, 'routes', layer.route(torch.cat((h, anchor), -1))
                            .reshape(batch, n, 3, c.relations))
            r_first, r_second, r_op = [routes[:, :, i] for i in range(3)]
            ef, es, op = [layer.relation(r) for r in (r_first, r_second, r_op)]
            first_state = torch.cat((h, anchor, ef), -1)
            addr1 = choose(ell, 'query_first', layer.query_first(first_state))
            rank1 = choose(ell, 'rank_first', layer.rank_first(first_state))
            write_rel, write_address = wr.detach().cpu().tolist(), wa.detach().cpu().tolist()
            indexes = [CausalIndex(c.candidates) for _ in range(batch)]
            for row in range(batch):
                for t in range(n):
                    if bool(valid[row, t]):
                        indexes[row].append(t, write_rel[row][t], write_address[row][t])
                        metrics['writes'] += 1
            operand1 = read_postings(indexes, r_first, addr1, rank1)
            left = dereference(h, operand1)
            # Query 2 MUST depend on actual retrieved operand-1 content.
            second_state = torch.cat((h, anchor, left, es), -1)
            addr2 = choose(ell, 'query_second', layer.query_second(second_state))
            rank2 = choose(ell, 'rank_second', layer.rank_second(second_state))
            operand2 = read_postings(indexes, r_second, addr2, rank2)
            right = dereference(h, operand2)
            continuation = choose(ell, 'continuation', layer.continue_ref(
                torch.cat((h, left, right, op), -1)))
            choices = torch.stack((torch.full_like(refs, -1), refs, operand1, operand2), -1)
            refs = choices.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
            if c.variant == 'no_reference' or memory == 'disabled':
                refs = torch.full_like(refs, -1)
            refs = torch.where(valid, refs, -1)
            features = torch.cat((h, left, right, ef, es, op,
                                  (torch.stack((operand1, operand2), -1)>=0).to(h.dtype)), -1)
            h = layer.norm(h + layer.compose(features)) * valid.unsqueeze(-1)
            metrics['operator_executions'] += int(valid.sum())
            ref_history.append(refs)
            operand_history.append(torch.stack((operand1, operand2), -1))
        return Output(self.lm_head(h), decisions, policies, used_noise,
                      ref_history, operand_history, metrics)


class StreamingMultiHop(StreamingInference):
    """Cached greedy decoder for SerialMultiHopLM; bulk prefill is inherited."""
    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 1:
            raise ValueError('step requires [batch]')
        if (tokens <= 0).any() or (tokens >= self.config.vocab).any():
            raise ValueError('step requires non-PAD in-vocabulary values')
        if self.cache is None:
            self.reset(tokens.shape[0], tokens.device)
        cache, model, cfg = self.cache, self.model, self.config
        if tokens.shape[0] != cache.batch_size:
            raise ValueError('batch changed')
        b, t = cache.batch_size, cache.position
        emb = model.embedding(tokens)
        k = cfg.local_kernel
        window = list(cache.token_embeds)
        if len(window)+1 < k:
            window = ([emb.new_zeros(emb.shape) for _ in range(k-1-len(window))]
                      + window)
        h = model.local_norm(emb + F.gelu(F.conv1d(
            torch.stack(window + [emb], -1), model.local.weight,
            model.local.bias).squeeze(-1)))
        cache.token_embeds.append(emb)
        refs = torch.full((b,), -1, device=tokens.device, dtype=torch.long)
        for ell, layer in enumerate(model.layers):
            states = cache.states[ell]
            anchor = self._deref(states, refs, h)
            cache.count['reference_dereferences'] += int((refs>=0).sum())
            wr, wa = layer.write_relation(h).argmax(-1), layer.write_address(h).argmax(-1)
            routes = layer.route(torch.cat((h,anchor), -1)).reshape(b,3,cfg.relations).argmax(-1)
            r1, r2, ro = [routes[:,i] for i in range(3)]
            ef, es, op = [layer.relation(x) for x in (r1,r2,ro)]
            first_state = torch.cat((h,anchor,ef),-1)
            addr1 = layer.query_first(first_state).argmax(-1)
            rank1 = layer.rank_first(first_state).argmax(-1)
            for row,(r,a) in enumerate(zip(wr.tolist(),wa.tolist())):
                cache.postings[ell][row].append(t,r,a)
                cache.count['writes'] += 1

            def select(relation, address, rank):
                ret = torch.full((b,), -1, device=h.device, dtype=torch.long)
                for row,(r,a,k) in enumerate(zip(relation.tolist(),address.tolist(),rank.tolist())):
                    hit = cache.postings[ell][row].lookup(r,a,t)
                    cache.count['reads'] += 1
                    cache.count['candidates'] += len(hit.candidates)
                    cache.count['omitted'] += hit.omitted
                    if k < len(hit.candidates):
                        ret[row] = hit.candidates[k]
                    else:
                        cache.count['null_reads'] += 1
                return ret

            id1 = select(r1, addr1, rank1)
            left = self._deref(states, id1, h)
            second_state = torch.cat((h,anchor,left,es),-1)
            addr2 = layer.query_second(second_state).argmax(-1)
            rank2 = layer.rank_second(second_state).argmax(-1)
            id2 = select(r2, addr2, rank2)
            right = self._deref(states,id2,h)
            cont = layer.continue_ref(torch.cat((h,left,right,op),-1)).argmax(-1)
            options = torch.stack((torch.full_like(refs,-1),refs,id1,id2),-1)
            new_refs = options.gather(-1,cont[:,None]).squeeze(-1)
            if cfg.variant == 'no_reference':
                new_refs = torch.full_like(refs,-1)
            features = torch.cat((h,left,right,ef,es,op,
                     torch.stack((id1,id2),-1).ge(0).to(h.dtype)), -1)
            next_h = layer.norm(h+layer.compose(features))
            cache.count['operator_executions'] += b
            states.append(h)
            refs = new_refs
            h = next_h
        cache.position += 1
        return model.lm_head(h)
