"""LRC-V4: page-addressable two-hop model, full reference + exact cached decoder.

Architecture hypothesis: access to historical occurrence page + local slot removes
V3's hard recency cutoff without dense prefix scoring. This does NOT establish
that the learned page logits correctly find distant facts. Both references use
identical causal semantics. The page policy is binary-coded; high inactive bits
are masked by available page count and out-of-range choices yield NULL.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .base.model import Config, Output, dereference
from .base.expressive_model import ProgramLayer, SerialMultiHopLM
from .base.fast_inference import StreamingInference
from .base.hierarchical_index import PagedCausalIndex


@dataclass
class PagedOutput(Output):
    active_page_bits: dict


class PagedProgramLayer(ProgramLayer):
    def __init__(self, cfg: Config, page_bits: int):
        super().__init__(cfg)
        d = cfg.width
        self.page_first = nn.Linear(3*d, 2*page_bits)
        self.page_second = nn.Linear(4*d, 2*page_bits)


class GatedMultiHopLM(SerialMultiHopLM):
    """Two dependent reads with arbitrary old-page selection, if correctly routed."""
    def __init__(self, config: Config = Config(), *, page_bits: int = 16, mode="learned"):
        if not 1 <= page_bits <= 30:
            raise ValueError('page_bits must be 1..30 (signed-64 safe)')
        super().__init__(config)
        if mode not in ("learned","always_on","always_off"): raise ValueError(mode)
        self.mode = mode
        self.fixed_gates = {0: 1}
        self.page_bits = page_bits
        self.layers = nn.ModuleList(PagedProgramLayer(config, page_bits)
                                    for _ in range(config.layers))
        self.read_gates = nn.ModuleList(nn.Linear(config.width,2) for _ in range(config.layers))

    def forward(self, tokens, *, sample=False, replay=None, overrides=None,
                noise=None, memory='normal'):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError('expected nonempty [batch, time] tokens')
        if (tokens < 0).any() or (tokens >= self.config.vocab).any():
            raise ValueError('out-of-vocabulary token')
        if memory not in ('normal','disabled'):
            raise ValueError('unsupported memory mode')
        valid = tokens != 0
        if ((~valid[:,:-1]) & valid[:,1:]).any():
            raise ValueError('padding must be a suffix')
        batch, n = tokens.shape
        cfg = self.config
        overrides = dict(overrides or {})
        if self.mode != "learned":
            for ell in range(cfg.layers): overrides.setdefault((ell,"read_gate"), torch.full_like(tokens,int(self.mode=="always_on")))
        if self.mode == "learned":
            for ell,value in self.fixed_gates.items():
                if ell not in range(cfg.layers) or value not in (0,1): raise ValueError("invalid fixed gate")
                overrides.setdefault((ell,"read_gate"),torch.full_like(tokens,value))
        policies, decisions, used_noise = {}, {}, {}
        active_bits = {}
        metrics = dict(writes=0, reads=0, candidates=0, omitted=0,
                       null_reads=0, reference_dereferences=0, operator_executions=0,
                       older_page_reads=0, invalid_page_reads=0,
                       active_page_bit_decisions=0, read_gate_open=0, read_gate_closed=0)
        refs_history, operand_history = [], []

        def choose(ell, name, logits):
            key = ell, name
            policies[key] = logits
            if key in overrides:
                action = overrides[key].to(tokens.device)
            elif replay is not None and key in replay:
                action = replay[key].to(tokens.device)
            elif sample:
                g = noise.get(key) if noise is not None else None
                if g is None:
                    u = torch.rand_like(logits).clamp_(1e-7,1-1e-7)
                    g = -(-u.log()).log()
                if g.shape != logits.shape:
                    raise ValueError('Gumbel noise shape mismatch')
                used_noise[key] = g
                action = (logits + g).argmax(-1)
            else:
                action = logits.argmax(-1)
            if (action.shape != logits.shape[:-1] or (action < 0).any()
                    or (action >= logits.shape[-1]).any()):
                raise ValueError(f'invalid action {key}')
            decisions[key] = action.detach().clone()
            return action

        def read_postings(indexes, rel, addr, rank, page_word, *, key):
            result = torch.full((batch,n), -1,device=tokens.device,dtype=torch.long)
            bits_mask = torch.zeros((batch,n,self.page_bits),device=tokens.device,dtype=torch.bool)
            relations, addresses, ranks, words = [v.detach().cpu().tolist()
                                    for v in (rel,addr,rank,page_word)]
            for row in range(batch):
                ix = indexes[row]
                for t in range(n):
                    if not bool(valid[row,t]) or not bool(open_gate[row,t]):
                        continue
                    pages = ix.page_count(relations[row][t],addresses[row][t],t)
                    width = min(self.page_bits, (pages-1).bit_length()) if pages else 0
                    bits_mask[row,t,:width] = True
                    metrics['active_page_bit_decisions'] += width
                    # Low bits define page; inactive high bits are ignored.
                    code = sum(int(v)<<bit for bit,v in enumerate(words[row][t][:width]))
                    hit = ix.lookup_page(relations[row][t],addresses[row][t],t,code)
                    metrics['reads'] += 1
                    metrics['candidates'] += len(hit.candidates)
                    metrics['omitted'] += hit.omitted
                    if hit.pages and hit.chosen_page is None:
                        metrics['invalid_page_reads'] += 1
                    if hit.chosen_page is not None and hit.chosen_page > 0:
                        metrics['older_page_reads'] += 1
                    if memory != 'disabled' and ranks[row][t] < len(hit.candidates):
                        result[row,t] = hit.candidates[ranks[row][t]]
                    else:
                        metrics['null_reads'] += 1
            active_bits[key] = bits_mask
            return result

        h = self.embedding(tokens)
        h = self.local_norm(h + F.gelu(self.local(F.pad(
            h.transpose(1,2), (cfg.local_kernel-1,0))).transpose(1,2)))
        h = h * valid.unsqueeze(-1)
        refs = torch.full_like(tokens,-1)
        for ell,layer in enumerate(self.layers):
            open_gate = choose(ell,"read_gate",self.read_gates[ell](h)).bool() & valid
            if memory == "disabled": open_gate = torch.zeros_like(open_gate)
            metrics["read_gate_open"] += int(open_gate.sum())
            metrics["read_gate_closed"] += int((valid & ~open_gate).sum())
            refs = torch.where(open_gate,refs,-1)
            anchor = dereference(h,refs)
            metrics['reference_dereferences'] += int((refs>=0).sum())
            wr = choose(ell,'write_relation',layer.write_relation(h))
            wa = choose(ell,'write_address',layer.write_address(h))
            routes = choose(ell,'routes',layer.route(torch.cat((h,anchor),-1))
                            .reshape(batch,n,3,cfg.relations))
            r1,r2,ro = [routes[:,:,k] for k in range(3)]
            ef,es,op = [layer.relation(x) for x in (r1,r2,ro)]
            first = torch.cat((h,anchor,ef),-1)
            a1 = choose(ell,'query_first',layer.query_first(first))
            k1 = choose(ell,'rank_first',layer.rank_first(first))
            p1 = choose(ell,'page_first',layer.page_first(first).reshape(
                batch,n,self.page_bits,2))
            wr_cpu,wa_cpu = wr.detach().cpu().tolist(),wa.detach().cpu().tolist()
            indexes = [PagedCausalIndex(cfg.candidates) for _ in range(batch)]
            for row in range(batch):
                for t in range(n):
                    if bool(valid[row,t]):
                        indexes[row].append(t,wr_cpu[row][t],wa_cpu[row][t])
                        metrics['writes'] += 1
            j1 = read_postings(indexes,r1,a1,k1,p1,key=(ell,'page_first'))
            left = dereference(h,j1)
            second = torch.cat((h,anchor,left,es),-1)
            a2 = choose(ell,'query_second',layer.query_second(second))
            k2 = choose(ell,'rank_second',layer.rank_second(second))
            p2 = choose(ell,'page_second',layer.page_second(second).reshape(
                batch,n,self.page_bits,2))
            j2 = read_postings(indexes,r2,a2,k2,p2,key=(ell,'page_second'))
            right = dereference(h,j2)
            continuation = choose(ell,'continuation',layer.continue_ref(
                torch.cat((h,left,right,op),-1)))
            opts = torch.stack((torch.full_like(refs,-1),refs,j1,j2),-1)
            refs = opts.gather(-1,continuation.unsqueeze(-1)).squeeze(-1)
            if cfg.variant == 'no_reference' or memory == 'disabled':
                refs = torch.full_like(refs,-1)
            refs = torch.where(valid & open_gate, refs,-1)
            features = torch.cat((h,left,right,ef,es,op,
                         (torch.stack((j1,j2),-1)>=0).to(h.dtype)),-1)
            h = layer.norm(h+layer.compose(features)) * valid.unsqueeze(-1)
            metrics['operator_executions'] += int(valid.sum())
            refs_history.append(refs)
            operand_history.append(torch.stack((j1,j2),-1))
        return PagedOutput(self.lm_head(h),decisions,policies,used_noise,
                           refs_history,operand_history,metrics,active_bits)


class StreamingGatedMultiHop(StreamingInference):
    """Cached decoder and bulk-prefill adapter with exact V4 page semantics."""
    def reset(self, batch_size: int, device=None):
        super().reset(batch_size,device)
        cfg = self.config
        self.cache.count.update(older_page_reads=0, invalid_page_reads=0,
                                active_page_bit_decisions=0, read_gate_open=0, read_gate_closed=0)
        self.cache.postings = [[PagedCausalIndex(cfg.candidates)
                                for _ in range(batch_size)]
                               for _ in range(cfg.layers)]
        return self

    @torch.no_grad()
    def step(self,tokens:torch.Tensor, *, gate_overrides=None)->torch.Tensor:
        if tokens.ndim!=1:
            raise ValueError('step requires [batch]')
        if (tokens<=0).any() or (tokens>=self.config.vocab).any():
            raise ValueError('non-padding in-vocabulary tokens required')
        if self.cache is None:
            self.reset(tokens.shape[0],tokens.device)
        cache,model,cfg = self.cache,self.model,self.config
        if len(tokens)!=cache.batch_size:
            raise ValueError('batch changed')
        b,t = cache.batch_size,cache.position
        emb = model.embedding(tokens)
        window = list(cache.token_embeds)
        if len(window)+1 < cfg.local_kernel:
            window = [emb.new_zeros(emb.shape) for _ in range(cfg.local_kernel-1-len(window))]+window
        h = model.local_norm(emb+F.gelu(F.conv1d(torch.stack(window+[emb],-1),
                        model.local.weight, model.local.bias).squeeze(-1)))
        cache.token_embeds.append(emb)
        refs = torch.full((b,),-1,dtype=torch.long,device=tokens.device)
        for ell,layer in enumerate(model.layers):
            states=cache.states[ell]
            if gate_overrides is not None and ell in gate_overrides:
                action=gate_overrides[ell].to(tokens.device)
                if action.shape!=(b,) or ((action<0)|(action>1)).any(): raise ValueError("invalid stream gate")
                opened=action.bool()
            elif model.mode!="learned": opened=torch.full((b,),model.mode=="always_on",device=tokens.device,dtype=torch.bool)
            elif ell in model.fixed_gates:
                value=model.fixed_gates[ell]
                if value not in (0,1): raise ValueError("invalid fixed gate")
                opened=torch.full((b,),bool(value),device=tokens.device,dtype=torch.bool)
            else: opened=model.read_gates[ell](h).argmax(-1).bool()
            cache.count["read_gate_open"]+=int(opened.sum())
            cache.count["read_gate_closed"]+=int((~opened).sum())
            refs=torch.where(opened,refs,-1)
            active=opened.nonzero().flatten()
            anchor=self._deref(states,refs,h)
            cache.count['reference_dereferences']+=int((refs>=0).sum())
            wr,wa=layer.write_relation(h).argmax(-1),layer.write_address(h).argmax(-1)
            routes=layer.route(torch.cat((h,anchor),-1)).reshape(b,3,cfg.relations).argmax(-1)
            r1,r2,ro=[routes[:,i] for i in range(3)]
            ef,es,op=[layer.relation(x) for x in (r1,r2,ro)]
            first=torch.cat((h,anchor,ef),-1)
            def head(module,features,bits=False):
                # Constant-open layers take the ordinary unmasked policy path.
                if len(active)==b:
                    logits=module(features)
                    if bits: logits=logits.reshape(b,model.page_bits,2)
                    return logits.argmax(-1)
                shape=(b,model.page_bits) if bits else (b,)
                out=torch.zeros(shape,dtype=torch.long,device=tokens.device)
                if len(active):
                    logits=module(features[active])
                    if bits: logits=logits.reshape(len(active),model.page_bits,2)
                    out[active]=logits.argmax(-1)
                return out
            a1=head(layer.query_first,first)
            k1=head(layer.rank_first,first)
            p1=head(layer.page_first,first,True)
            for row,(r,a) in enumerate(zip(wr.tolist(),wa.tolist())):
                cache.postings[ell][row].append(t,r,a)
                cache.count['writes']+=1

            def select(rel,addr,rank,bits):
                out=torch.full((b,),-1,dtype=torch.long,device=tokens.device)
                for row,(r,a,k,word) in enumerate(zip(rel.tolist(),addr.tolist(),
                                                       rank.tolist(),bits.tolist())):
                    if not bool(opened[row]): continue
                    ix=cache.postings[ell][row]
                    pages=ix.page_count(r,a,t)
                    width=min(model.page_bits,(pages-1).bit_length()) if pages else 0
                    cache.count['active_page_bit_decisions']+=width
                    page=sum(int(bit)<<i for i,bit in enumerate(word[:width]))
                    hit=ix.lookup_page(r,a,t,page)
                    cache.count['reads']+=1
                    cache.count['candidates']+=len(hit.candidates)
                    cache.count['omitted']+=hit.omitted
                    if hit.pages and hit.chosen_page is None:
                        cache.count['invalid_page_reads']+=1
                    if hit.chosen_page is not None and hit.chosen_page>0:
                        cache.count['older_page_reads']+=1
                    if k<len(hit.candidates):
                        out[row]=hit.candidates[k]
                    else:
                        cache.count['null_reads']+=1
                return out

            j1=select(r1,a1,k1,p1)
            left=self._deref(states,j1,h)
            second=torch.cat((h,anchor,left,es),-1)
            a2=head(layer.query_second,second)
            k2=head(layer.rank_second,second)
            p2=head(layer.page_second,second,True)
            j2=select(r2,a2,k2,p2)
            right=self._deref(states,j2,h)
            continuation=head(layer.continue_ref,torch.cat((h,left,right,op),-1))
            opts=torch.stack((torch.full_like(refs,-1),refs,j1,j2),-1)
            newrefs=opts.gather(-1,continuation[:,None]).squeeze(-1)
            newrefs=torch.where(opened,newrefs,-1)
            if cfg.variant=='no_reference':
                newrefs=torch.full_like(refs,-1)
            features=torch.cat((h,left,right,ef,es,op,
                               (torch.stack((j1,j2),-1)>=0).to(h.dtype)),-1)
            next_h=layer.norm(h+layer.compose(features))
            states.append(h)
            cache.count['operator_executions']+=b
            refs=newrefs
            h=next_h
        cache.position+=1
        return model.lm_head(h)


    @torch.no_grad()
    def prefill(self,tokens):
        """Vectorized gated forward plus writer-state/index capture; resets cache.

        Inherited temporary hooks are non-reentrant for a shared model. Greedy
        normal inference only; use incremental reference for differential tests.
        """
        return super().prefill(tokens)


class LocalOnlyStream:
    """Same-weight memory-disabled local path; no posting/history allocation.

    Raw logits, not log probabilities. Only valid for all-layer memory off.
    Uses exact finite local receptive field plus retained relation/composer path.
    """
    def __init__(self,model):
        if model.training: raise ValueError("eval model required")
        self.model=model;self.reset()
    def reset(self,batch_size=None,device=None):
        from collections import deque
        self.embeds=deque(maxlen=max(0,self.model.config.local_kernel-1));self.position=0;self.batch_size=batch_size
        return self
    @torch.no_grad()
    def step(self,tokens):
        m=self.model;c=m.config
        if tokens.ndim!=1 or (tokens<=0).any() or (tokens>=c.vocab).any(): raise ValueError("valid unpadded B tokens required")
        if self.batch_size is not None and len(tokens)!=self.batch_size: raise ValueError("batch changed")
        self.batch_size=len(tokens);e=m.embedding(tokens);prior=list(self.embeds)
        prior=[torch.zeros_like(e)]*(c.local_kernel-1-len(prior))+prior
        h=m.local_norm(e+F.gelu(F.conv1d(torch.stack(prior+[e],-1),m.local.weight,m.local.bias).squeeze(-1)));self.embeds.append(e)
        for layer in m.layers:
            z=torch.zeros_like(h);routes=layer.route(torch.cat((h,z),-1)).reshape(len(tokens),3,c.relations).argmax(-1)
            ef,es,op=[layer.relation(routes[:,j]) for j in range(3)]
            h=layer.norm(h+layer.compose(torch.cat((h,z,z,ef,es,op,h.new_zeros((len(tokens),2))),-1)))
        self.position+=1;return m.lm_head(h)
    @torch.no_grad()
    def prefill(self,tokens):
        c=self.model.config
        if tokens.ndim!=2 or tokens.shape[1]<1 or (tokens<=0).any() or (tokens>=c.vocab).any(): raise ValueError("valid nonempty unpadded B,T required")
        self.reset(tokens.shape[0])
        for t in range(max(0,tokens.shape[1]-c.local_kernel),tokens.shape[1]): out=self.step(tokens[:,t])
        self.position=tokens.shape[1];return out
