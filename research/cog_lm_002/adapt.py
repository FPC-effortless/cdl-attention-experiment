import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from architecture import Readout, backbone, features, successor_ids
from data import corpus


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def train(a):
    torch.set_num_threads(a.threads)
    m = backbone(a.weights)
    tokens, bounds = corpus(a.data, 'train')
    starts = np.concatenate([np.arange(lo, hi-128, 128, dtype=np.int64) for lo, hi in zip(bounds, bounds[1:])])
    order = np.random.default_rng(20261121).choice(starts, 12288, replace=False).reshape(-1, 8)
    names = ['mamba_lookup'] if m.c.arm == 'mamba2' else ['cog_lookup', 'cog_revision']
    heads, opts, curves = {}, {}, {}
    for name in names:
        torch.manual_seed(17)
        heads[name] = Readout(m.c.width, name == 'cog_revision')
        opts[name] = torch.optim.AdamW(heads[name].parameters(), lr=.001, foreach=False)
        curves[name] = []
    start = time.perf_counter()
    for step, offsets in enumerate(order):
        seq = torch.from_numpy(tokens[offsets[:, None] + np.arange(129)].astype('int64'))
        x, y = seq[:, :-1], seq[:, 1:]
        with torch.no_grad():
            h, _ = features(m, x)
            index = successor_ids(x, m.c.capacity)
        for name in names:
            opt = opts[name]
            opt.param_groups[0]['lr'] = .001 * min(1, (step+1)/32) * (.2 + .4*(1+math.cos(math.pi*step/len(order))))
            pred, _ = heads[name](h, index, m.emb.weight)
            loss = F.nll_loss(pred.flatten(0, 1), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 1., error_if_nonfinite=True)
            opt.step()
            curves[name].append(float(loss.detach()))
        if (step+1)%64 == 0 or step == 0:
            print(step+1, [round(curves[n][-1], 3) for n in names], round(time.perf_counter()-start, 1), flush=True)
    a.out.mkdir(exist_ok=True, parents=True)
    for name, head in heads.items():
        path = a.out / (name + '.pt')
        torch.save(head.state_dict(), path)
        (a.out / (name + '.json')).write_text(json.dumps(dict(parent=a.weights.name,
            parent_sha256=digest(a.weights), adapter_sha256=digest(path), name=name, seed=17,
            steps=len(order), additional_targets=order.size*128, total_targets=524288+order.size*128,
            offsets_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
            trainable=sum(p.numel() for p in head.parameters()), base=sum(p.numel() for p in m.parameters()),
            combined_wall_s=time.perf_counter()-start, loss=curves[name]))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('weights','data','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    train(parser.parse_args())
