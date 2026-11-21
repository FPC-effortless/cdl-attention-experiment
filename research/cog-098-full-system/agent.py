"""Shared cross-domain policy, bounded working memory and observed-transition planner."""
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'cog-071-080'), str(HERE.parent / 'cog-097e')]
from boundary import Workspace
from experiment import D, V, RawPolicy, ids, tokenize


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class AttentionPolicy(nn.Module):
    """Self-attention across the same hashed subwords, pooled in blocks of 12."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * D + 8, 64), nn.GELU(), nn.Linear(64, 1))

    def blocks(self, texts):
        rows = [ids(text) for text in texts]
        width = 12 * ((max(map(len, rows)) + 11) // 12)
        tokens = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, row in enumerate(rows):
            tokens[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = True
        valid = mask.reshape(len(rows), -1, 12).sum(2)
        packed = (self.emb(tokens) * mask.unsqueeze(-1)).reshape(len(rows), -1, 12, D)
        return packed.sum(2) / valid.clamp_min(1).unsqueeze(-1), valid > 0

    def forward(self, queries, choices, features):
        q, qm = self.blocks(queries)
        a, am = self.blocks(choices)
        seq = torch.cat([q, a], dim=1)
        mask = torch.cat([qm, am], dim=1)
        out, _ = self.attn(seq, seq, seq, key_padding_mask=~mask, need_weights=False)
        q, a = (out + seq).split([q.size(1), a.size(1)], dim=1)
        q = (q * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True)
        a = (a * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True)
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        return self.head(torch.cat([q, a, (q-a).abs(), x], dim=1)).flatten()


class FullCOGAgent:
    def __init__(self, arm='cog', seed=1701, memory=True):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.arm, self.seed, self.memory = arm, seed, memory
        self.assisted = True
        self.rng = random.Random(seed)
        self.policy = AttentionPolicy() if arm == 'attention' else RawPolicy()
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=.002)
        self.graph = {}
        self.updates = 0
        self.frozen = False
        self.reset()

    def reset(self):
        self.workspace = Workspace()
        self.version = 0
        self.history = []
        self.visits = Counter()
        self.last = None
        self.events = []
        self.edges = []
        self.pending = None

    def observe(self, text, legal, priors):
        if not legal or len(legal) != len(set(legal)) or len(priors) != len(legal):
            raise ValueError('invalid public action menu')
        if self.pending is not None:
            raise RuntimeError('execute or cancel the staged proposal first')
        self.version += 1
        self.state = digest(text)
        self.legal = tuple(legal)
        self.priors = priors
        self.query = text
        if self.memory and self.history:
            self.query = text + ' prior observations ' + ' '.join(reversed(self.history[-3:]))
        self.history = (self.history + [text])[-3:]

    def expected(self, state, action, depth):
        row = self.graph.get(state, {}).get(action)
        if not self.memory or self.arm == 'heuristic' or not row:
            return 0.
        visits, total, futures = row
        value = total / visits
        if depth:
            for child, count in futures.items():
                value += .8 * count / visits * max(
                    (self.expected(child, a, depth-1) for a in self.graph.get(child, {})), default=0.)
        return min(value, 1.)

    def propose(self):
        if self.pending is not None:
            raise RuntimeError('proposal already staged')
        rows = []
        words = set(tokenize(self.query))
        for action, prior in zip(self.legal, self.priors):
            other = set(tokenize(action))
            rows.append([prior, len(words & other)/max(1, len(other)),
                         self.expected(self.state, action, 1),
                         min(1., self.visits[self.state, action]/3) if self.memory else 0.,
                         min(len(words)/100, 1.), min(len(other)/40, 1.),
                         min(self.version/48, 1.), 1.])
        if self.arm == 'heuristic':
            scores = np.array(self.priors)
        else:
            with torch.no_grad():
                logits = self.policy([self.query]*len(rows), self.legal, rows)
                scores = torch.sigmoid(logits).numpy() + .3*np.array([x[2] for x in rows])
        i = int(np.argmax(scores))
        if not self.frozen and self.arm != 'heuristic' and self.rng.random() < .35:
            i = self.rng.randrange(len(rows))
        action = self.legal[i]
        self.workspace.stage('observation', (), self.version)
        self.workspace.stage('candidate', ('observation',), (self.version, action))
        self.pending = self.query, action, rows[i]
        return action

    def act(self, environment):
        version, action = self.workspace.prepared('candidate')
        self.last = None
        try:
            if version != self.version or action not in self.legal:
                raise RuntimeError('stale or illegal proposal')
            outcome = environment.step(action)
            self.last = self.pending
        finally:
            self.workspace.discard('observation')
            self.workspace.undo.clear()
            self.pending = None
        self.visits[self.state, action] += 1
        return outcome

    def learn(self, reward, next_observation, terminal):
        if self.frozen or self.last is None:
            raise RuntimeError('updates require one executed training action')
        q, a, x = self.last
        self.last = None
        self.events.append((q, a, x, float(reward)))
        self.edges.append((self.state, a, None if terminal else digest(next_observation), float(reward)))

    def finish(self, delayed=False):
        if self.frozen:
            raise RuntimeError('frozen')
        if not self.events:
            return
        for state, action, child, reward in self.edges:
            row = self.graph.setdefault(state, {}).setdefault(action, [0, 0., Counter()])
            row[0] += 1
            row[1] += reward
            if child is not None:
                row[2][child] += 1
        q, a, x, y = zip(*self.events)
        y = list(y)
        if delayed:
            for i in range(len(y)-2, -1, -1):
                y[i] = max(y[i], .95*y[i+1])
        logits = self.policy(list(q), list(a), list(x))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y),
                                                              pos_weight=torch.tensor(3.))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        self.events.clear()
        self.edges.clear()

    def freeze(self):
        self.reset()
        self.policy.eval()
        self.frozen = True

    def snapshot(self):
        if not self.frozen:
            raise RuntimeError('only frozen episode-boundary checkpoints supported')
        graph = {s: {a: [n, r, dict(f)] for a, (n, r, f) in actions.items()}
                 for s, actions in self.graph.items()}
        return dict(arm=self.arm, seed=self.seed, weights=self.policy.state_dict(),
                    graph=graph, updates=self.updates, assisted=self.assisted)

    @classmethod
    def restore(cls, snapshot, memory=True):
        agent = cls(snapshot['arm'], snapshot['seed'], memory)
        agent.policy.load_state_dict(snapshot['weights'])
        agent.assisted = snapshot.get('assisted',True)
        agent.graph = snapshot['graph']
        agent.updates = snapshot['updates']
        agent.freeze()
        return agent

    def explain_provenance(self):
        return dict(state=self.state, version=self.version, menu=digest(repr(self.legal)),
                    updates=self.updates, memory=self.memory)
