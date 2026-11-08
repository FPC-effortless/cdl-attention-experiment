"""VB-001: prospective, structured-template versioned-memory baselines.

No gold support enters model.forward. Record boundaries are supplied to ALL arms.
This is a reconstructed task, not a replay of the unavailable LRC versioned task.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

NAMES = ('Alice', 'Boris', 'Cora', 'Dion')
PLACES = ('Rome', 'Lima', 'Oslo', 'Accra')
ERAS = ('first', 'second')
WORDS = ('<pad>', 'During', 'the', 'era', 'partnered', 'with', 'lived', 'in', '.',
         'where', 'did', "'s", 'partner', 'live', '?') + NAMES + PLACES + ERAS
VOCAB = {word: i for i, word in enumerate(WORDS)}
MAX_WORDS = 12
N_RECORDS = 16
ARMS = ('transformer_answer', 'transformer_support', 'memnet_answer', 'memnet_support', 'query_only')

@dataclass(frozen=True)
class Config:
    seeds: tuple = (301, 302, 303)
    steps: int = 2000
    batch: int = 32
    train_groups: int = 2048
    test_groups: int = 256
    lr: float = 0.002
    support_weight: float = 1.0
    train_seed: int = 71001
    iid_seed: int = 71002
    heldout_seed: int = 71003
    width_transformer: int = 32
    width_memnet: int = 51
    width_query: int = 32


def encode(words):
    ids = [VOCAB[x] for x in words]
    assert len(ids) <= MAX_WORDS
    return ids + [0] * (MAX_WORDS - len(ids))


def make_group(rng: random.Random, heldout: bool = False):
    """Four counterfactual twins differing ONLY in the relevant location value.

    Full snapshot has one partner and one location per entity in each era.
    Query follows partner then location at the specified era, including old eras.
    Held-out compositional pairs are (entity, its next cyclic entity).
    """
    entity, era = rng.randrange(4), rng.randrange(2)
    excluded = (entity + 1) % 4
    target_person = excluded if heldout else rng.choice([x for x in range(4) if x != excluded])
    facts = []
    for e in range(2):
        block = [(e, n, rel, rng.randrange(4)) for n in range(4) for rel in range(2)]
        rng.shuffle(block)
        facts.extend(block)
    first = next(i for i, f in enumerate(facts) if f[:3] == (era, entity, 0))
    facts[first] = (era, entity, 0, target_person)
    second = next(i for i, f in enumerate(facts) if f[:3] == (era, target_person, 1))
    question = encode(['During', 'the', ERAS[era], 'era', 'where', 'did', NAMES[entity], "'s", 'partner', 'live', '?'])
    examples = []
    for answer in range(4):
        variant = list(facts)
        variant[second] = (era, target_person, 1, answer)
        records = []
        for e, n, rel, val in variant:
            words = ['During', 'the', ERAS[e], 'era', NAMES[n]]
            words += ['partnered', 'with', NAMES[val]] if rel == 0 else ['lived', 'in', PLACES[val]]
            records.append(encode(words + ['.']))
        examples.append(dict(records=records, query=question, answer=answer,
                             support=[first, second], era=era, entity=entity, person=target_person))
    return examples


def dataset(seed, groups, heldout=False):
    rng = random.Random(seed)
    examples = [item for _ in range(groups) for item in make_group(rng, heldout)]
    return {k: torch.tensor([e[k] for e in examples], dtype=torch.long) for k in examples[0]}


def digest_data(data):
    h = hashlib.sha256()
    for key in sorted(data):
        h.update(key.encode())
        h.update(data[key].numpy().tobytes())
    return h.hexdigest()


def take(data, idx):
    return {k: v[idx] for k, v in data.items()}


class SentenceEncoder(nn.Module):
    """Shared design, independently learned per model; no semantic parsing.

    Learned word-position weights distinguish subject from object. Explicit
    sentence boundaries and finite token vocabulary are privileged scaffolding.
    """
    def __init__(self, d):
        super().__init__()
        self.word = nn.Embedding(len(WORDS), d, padding_idx=0)
        self.position = nn.Parameter(torch.ones(MAX_WORDS, d) + 0.02 * torch.randn(MAX_WORDS, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        emb = self.word(tokens) * self.position
        count = tokens.ne(0).sum(-1, keepdim=True).clamp_min(1).sqrt()
        return self.norm(emb.sum(-2) / count)


class AttentionLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))

    def forward(self, x, hard=False):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        length = x.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        # Query must read a memory record, rather than itself. Same candidate set
        # as MemNet at the query. Records keep causal self attention.
        mask[-1, -1] = True
        scores = scores.masked_fill(mask, float('-inf'))
        weights = scores.softmax(-1)
        query_weights = weights[:, -1, :-1]
        if hard:
            weights = weights.clone()
            weights[:, -1, :] = F.one_hot(weights[:, -1, :].argmax(-1), length).to(weights.dtype)
        x = x + self.out(weights @ v)
        return x + self.ff(self.norm2(x)), query_weights


class TinyTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = SentenceEncoder(d)
        self.time = nn.Embedding(N_RECORDS+1, d)
        self.layers = nn.ModuleList([AttentionLayer(d) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))

    def forward(self, records, query, hard=False):
        x = torch.cat([self.encoder(records), self.encoder(query).unsqueeze(1)], 1)
        x = x + self.time(torch.arange(x.shape[1], device=x.device))
        reads = []
        for layer in self.layers:
            x, read = layer(x, hard)
            reads.append(read)
        return self.head(x[:, -1]), torch.stack(reads, 1)


class MemoryNetwork(nn.Module):
    """Two-hop end-to-end memory network adaptation, untied hop projections."""
    def __init__(self, d):
        super().__init__()
        self.encoder = SentenceEncoder(d)
        self.time = nn.Embedding(N_RECORDS, d)
        self.keys = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.values = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.head = nn.Linear(d, 4)

    def forward(self, records, query, hard=False):
        memory = self.encoder(records) + self.time(torch.arange(records.shape[1], device=records.device))
        state = self.encoder(query)
        reads = []
        for key, value, update, norm in zip(self.keys, self.values, self.updates, self.norms):
            weights = torch.einsum('bd,bnd->bn', state, key(memory)) / math.sqrt(state.shape[-1])
            weights = weights.softmax(-1)
            reads.append(weights)
            used = F.one_hot(weights.argmax(-1), weights.shape[-1]).to(weights.dtype) if hard else weights
            read = torch.einsum('bn,bnd->bd', used, value(memory))
            state = norm(update(state) + read)
        return self.head(state), torch.stack(reads, 1)


class QueryOnly(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = SentenceEncoder(d)
        self.head = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, 4))

    def forward(self, records, query, hard=False):
        return self.head(self.encoder(query)), None


def build(arm, config):
    if arm.startswith('transformer'):
        return TinyTransformer(config.width_transformer)
    if arm.startswith('memnet'):
        return MemoryNetwork(config.width_memnet)
    return QueryOnly(config.width_query)


def objective(model, batch, support_weight):
    logits, reads = model(batch['records'], batch['query'])
    answer = F.cross_entropy(logits, batch['answer'])
    auxiliary = logits.new_zeros(())
    if support_weight:
        auxiliary = -reads.gather(-1, batch['support'].unsqueeze(-1)).clamp_min(1e-9).log().mean()
    return answer + support_weight * auxiliary, answer, auxiliary


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare(out, config):
    out.mkdir(parents=True, exist_ok=False)
    train = dataset(config.train_seed, config.train_groups)
    torch.save(train, out/'train.pt')
    meta = dict(config=asdict(config), arms=ARMS, source_sha256=source_hash(),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                torch=torch.__version__, python=platform.python_version(), platform=platform.platform(),
                threads=1, train_sha256=digest_data(train),
                parameters={a: sum(p.numel() for p in build(a, config).parameters()) for a in ARMS},
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def fit(out, config, arm, seed, train):
    torch.manual_seed(seed)
    model = build(arm, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    order = torch.Generator().manual_seed(100000 + seed)
    trace, order_hash = [], hashlib.sha256()
    start = time.perf_counter()
    weight = config.support_weight if arm.endswith('_support') else 0.0
    for step in range(1, config.steps+1):
        idx = torch.randint(len(train['answer']), (config.batch,), generator=order)
        order_hash.update(idx.numpy().tobytes())
        batch = take(train, idx)
        loss, answer, auxiliary = objective(model, batch, weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss: {arm}, {seed}, {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append(dict(step=step, total=float(loss.detach()), answer=float(answer.detach()), auxiliary=float(auxiliary.detach())))
    elapsed = time.perf_counter()-start
    receipt = dict(arm=arm, seed=seed, steps=config.steps, train_seconds=elapsed,
                   batch_order_sha256=order_hash.hexdigest(), trace=trace, source_sha256=source_hash())
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), order_rng=order.get_state(), receipt=receipt,
                    config=asdict(config)), out/f'{arm}_{seed}.pt')
    (out/f'{arm}_{seed}.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(event='fit_complete', arm=arm, seed=seed, seconds=elapsed, last=trace[-1])), flush=True)


@torch.inference_mode()
def evaluate(model, data, hard=False, ablation=None):
    preds, losses, reads_all = [], [], []
    for start in range(0, len(data['answer']), 128):
        batch = take(data, slice(start, start+128))
        records = batch['records']
        if ablation == 'zero':
            records = torch.zeros_like(records)
        logits, reads = model(records, batch['query'], hard=hard)
        preds.append(logits.argmax(-1))
        losses.append(F.cross_entropy(logits, batch['answer'], reduction='none'))
        if reads is not None:
            reads_all.append(reads.argmax(-1))
    pred, loss = torch.cat(preds), torch.cat(losses)
    correct = pred.eq(data['answer'])
    result = dict(accuracy=float(correct.float().mean()), nll=float(loss.mean()), n=len(pred),
                  quartet_accuracy=float(correct.reshape(-1, 4).all(-1).float().mean()),
                  old_accuracy=float(correct[data['era'].eq(0)].float().mean()),
                  latest_accuracy=float(correct[data['era'].eq(1)].float().mean()))
    if reads_all:
        selected = torch.cat(reads_all)
        matched = selected.eq(data['support'])
        result.update(hop1=float(matched[:, 0].float().mean()), hop2=float(matched[:, 1].float().mean()),
                      both_supports=float(matched.all(-1).float().mean()))
    return result, dict(pred=pred, loss=loss, selected=torch.cat(reads_all) if reads_all else None)


def test_all(out, config):
    # Enforce the all-fits-before-heldout boundary.
    for arm in ARMS:
        for seed in config.seeds:
            assert (out/f'{arm}_{seed}.pt').exists(), 'Incomplete fit matrix'
    datasets = dict(iid=dataset(config.iid_seed, config.test_groups),
                    heldout_pairs=dataset(config.heldout_seed, config.test_groups, True))
    train = torch.load(out/'train.pt', weights_only=True)
    # Exclude identical complete examples across train/evaluation.
    def signatures(d):
        return {tuple(torch.cat([r.flatten(), q]).tolist()) for r, q in zip(d['records'], d['query'])}
    train_set = signatures(train)
    assert all(not (train_set & signatures(d)) for d in datasets.values())
    torch.save(datasets, out/'evaluation_data.pt')
    results, predictions = [], {}
    for seed in config.seeds:
        for arm in ARMS:
            checkpoint = torch.load(out/f'{arm}_{seed}.pt', weights_only=True)
            model = build(arm, config).eval()
            model.load_state_dict(checkpoint['model'])
            item = dict(arm=arm, seed=seed)
            for name, data in datasets.items():
                item[name], predictions[f'{arm}/{seed}/{name}'] = evaluate(model, data)
                item[name+'_hard'], _ = evaluate(model, data, hard=True)
                item[name+'_memory_off'], _ = evaluate(model, data, ablation='zero')
            results.append(item)
    torch.save(predictions, out/'predictions.pt')
    result = dict(results=results, evaluation_sha256={k:digest_data(v) for k,v in datasets.items()},
                  complete_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'evaluate', 'counts'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = Config()
    if args.action == 'counts':
        print({a: sum(p.numel() for p in build(a, config).parameters()) for a in ARMS})
        return
    if args.action == 'run':
        print(json.dumps(prepare(args.out, config)), flush=True)
        train = torch.load(args.out/'train.pt', weights_only=True)
        for seed in config.seeds:
            for arm in ARMS:
                fit(args.out, config, arm, seed, train)
        test_all(args.out, config)
    else:
        test_all(args.out, config)

if __name__ == '__main__':
    main()
