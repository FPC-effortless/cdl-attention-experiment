"""Context substitution on the same real held-out next-byte targets."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trace_fault import load_corpus, loadstudy, mean
import math

@torch.inference_mode()
def evaluate(m, corpus):
    rng = np.random.default_rng(98723)
    losses = collections.defaultdict(list)
    counts = np.ones(256)
    for _ in range(16):
        counts += np.bincount(corpus.batch('train', rng, 128, 256).flatten(), minlength=256)
    prior = torch.from_numpy(counts / counts.sum())
    for _ in range(64):
        raw = torch.from_numpy(corpus.batch('dev', rng, 16, 96))
        donor = torch.from_numpy(corpus.batch('dev', rng, 16, 96))
        y = raw[:, 89:]
        for k in (0,4,16,64,89):
            x = torch.cat([donor[:, :89 - k], raw[:, 89 - k:96]], 1)
            z = m(x)[:, 88:]
            scores = F.cross_entropy(z.flatten(0, 1), y.flatten(), reduction='none').view(-1, 8) / math.log(2)
            losses[k].extend(scores.mean(dim=1).tolist())
        losses['unigram'].extend((-prior[y].log2().mean(dim=1)).tolist())
    orig = np.array(losses[89])
    return {k: {'bpb': mean(v), 'difference': mean(np.array(v) - orig), 'window_paired_se': float(np.std(np.array(v) - orig, ddof=1) / np.sqrt(len(v)))} for k, v in losses.items()}

if __name__=="__main__":
    a = sys.argv
    torch.set_num_threads(1)
    model, *_ = loadstudy(a[1])
    Path(a[2]).write_text(json.dumps(evaluate(model, load_corpus(Path(a[3]))), indent=2))
