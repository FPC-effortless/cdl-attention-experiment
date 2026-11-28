"""Track one edited fact through the recurrent state and through each layer."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

from trace_fault import ids, loadstudy, mean, probe

@torch.inference_mode()
def trajectory(m):
    rng = np.random.default_rng(700131)
    rows = []
    used = set()
    # Fixed target 'a' at the first position. Other labels are independent of a.
    while len(rows)<768:
        ctx = ''.join((chr(i) for i in rng.integers(97, 123, 5)))
        a, b = rng.choice(4, size=2, replace=False)
        others = rng.permutation(list('bcd'))
        follow = [f'!{k}{v}' for k, v in zip(others, rng.integers(4, size=3))]
        if ctx in used:
            continue
        used.add(ctx)
        rows.append((ctx, int(a), int(b), follow))
    feat = collections.defaultdict(list)
    effect = collections.defaultdict(list)
    routes = collections.defaultdict(list)
    for start in range(0,len(rows),16):
        group = rows[start:start + 16]
        state = None
        chunks = [[], [], [], [], [], [], []]
        for pre,a,b,suffix in group:
            for v in [a,b]:
                seq = [pre + '!a' + str(v), *suffix, '#a', '00', '=']
                for out, part in zip(chunks, seq):
                    out.append(part)
        for stage,text in enumerate(chunks):
            tokens = ids(text)
            m0 = m.parent
            h, _, ax = m0.local(tokens, m0.core.emb, None if state is None else state[1])
            if start>=384:
                routes[stage].extend(torch.stack([ax[0][:, -1].argmax(-1), ax[1][:, -1].argmax(-1)], 1).tolist())
            z, state, hidden = m(tokens, state=state, features=True)
            for ix,l in enumerate(state[0]):
                components = {'fast': l[0], 'protected': l[1][:, :8], 'background': l[1][:, 8:], 'slow': state[2][ix][0]}
                for name,v in components.items():
                    v = v.reshape(len(text), -1)
                    f = v.numpy()
                    feat[stage, ix, name].append(f)
                    if start >= 384:
                        effect[stage, ix, name].extend(np.linalg.norm(f[::2] - f[1::2], axis=1).tolist())
            feat[stage, -1, 'hidden'].append(hidden[:, -1].numpy())
    labs = np.array([k for _, a, b, _ in rows for k in (a, b)])
    s = 768
    out = {}
    for (stage,ix,name),values in feat.items():
        v = np.concatenate(values)
        y = probe(v[:s], v[s:], labs[:s], labs[s:])
        ones = y == labs[s:]
        ds = np.array(effect[stage, ix, name])
        out[f"{stage}/{ix}/{name}"]={"digit":mean(ones),"both":mean(ones.reshape(-1,2).all(1)),
            "fraction_exactly_identical":mean(ds==0) if len(ds) else None}
    return {"stages":["target digit","distractor 1","distractor 2","distractor 3","query key","two identity operations","answer delimiter"],"measured":out,
            "last_byte_role_counts":{i:np.bincount(np.array(x)[:,0],minlength=4).tolist() for i,x in routes.items()}}
if __name__=="__main__":
    torch.set_num_threads(1)
    m, *_ = loadstudy(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(trajectory(m), indent=2))
