"""Frozen COG assays. No probe or state intervention is part of the model."""
import argparse, collections, hashlib, io, itertools, json, sys, zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
for d in ("unified-search-002","cog-balance-001","cog-adaptive-repair-002"):
    sys.path.insert(0, str(HERE.parent / d))
from scale import build_model
from repair_model import build_repair_model
from corpus import load_corpus

def ids(xs):
    assert len(set(map(len, xs))) == 1
    return torch.tensor([list(x.encode()) for x in xs])

def mean(x):
    return float(np.mean(x))

def runs(model, xs, st=None):
    return model(ids(xs), state=st, features=True)

def get_state(state, c):
    if c == 'tail':
        return state[1].float()
    if c == 'slow':
        return torch.cat([v[0].flatten(1) for v in state[2]], 1)
    return torch.cat([(v[0] if c == 'fast' else v[1][:, :8] if c == 'protected' else v[1][:, 8:]).flatten(1) for v in state[0]], 1)

def patch(rec,don,part):
    out = []
    for (a,x),(b,y) in zip(rec[0],don[0]):
        out.append((b if part == 'fast' else a, torch.cat([y[:, :8] if part == 'protected' else x[:, :8], y[:, 8:] if part == 'background' else x[:, 8:]], 1)))
    return (out, don[1] if part == 'tail' else rec[1], don[2] if part == 'slow' else rec[2])

def probe(train,test,ys,yt):
    x = np.array(train, dtype=np.float64)
    z = np.array(test, dtype=np.float64)
    mu = x.mean(0)
    x = x - mu
    z = z - mu
    # Isotropic ridge, normalized by the total training feature variance.
    norm = np.sum(x * x) / max(1, min(x.shape))
    if norm < 1e-14:
        return np.full(len(z), np.bincount(ys, minlength=4).argmax())
    y = np.eye(4)[ys]
    prior = y.mean(0)
    y -= prior
    reg = 0.005 * norm
    if len(x)<x.shape[1]:
        mat = x @ x.T
        mat.flat[::len(mat) + 1] += reg
        scores = z @ x.T @ np.linalg.solve(mat, y)
    else:
        mat = x.T @ x
        mat.flat[::len(mat) + 1] += reg
        scores = z @ np.linalg.solve(mat, x.T @ y)
    return (scores + prior).argmax(1)

def cases(count,seed,split,domain,independent=False):
    rng = np.random.default_rng(seed)
    seen = set()
    names = ['Ada', 'Ben', 'Cia', 'Dan']
    while len(seen)<count:
        v = rng.permutation(4)
        order = rng.permutation(4)
        weak = rng.integers(4, size=4)
        seq = rng.permutation(8)
        if independent:
            v[2:] = rng.integers(4, size=2)
        def make(w):
            if domain=="symbol":
                strong = [f"!{'abcd'[i]}{w[i]}" for i in order]
                return ''.join(strong) + f'~d{weak[3]}'
            strong = [f'The log lists {names[i]} at {w[i]}. ' for i in range(4)]
            rumor = [f'A rumor puts {names[i]} at {weak[i]}. ' for i in range(4)]
            return ''.join(((strong + rumor)[i] for i in seq))
        w = v.copy()
        w[:2] = w[:2][::-1]
        a, b = (make(v), make(w))
        canon = sorted([a, b])[0]
        sig = hashlib.sha256(canon.encode()).digest()
        if (sig[0] % 4 == 0) != split or canon in seen:
            continue
        seen.add(canon)
        yield (a, b, int(v[0]), int(v[1]))

def assay(m,native,domain):
    train = list(cases(384, 31091, False, domain))
    dev = list(cases(192, 48153, True, domain))
    q = ['#a00=', '#b00='] if domain == 'symbol' else [f'What digit does the log give for {s}?' for s in ('Ada', 'Ben')]
    reps = collections.defaultdict(list)
    scores = []
    transports = collections.defaultdict(list)
    histroles = []
    queryroles = []
    fullerr = 0.0
    for start in range(0,len(train+dev),8):
        group = (train + dev)[start:start + 8]
        n = len(group)
        x = [r[j] for r in group for j in range(2)]
        _, state, _ = runs(m, x)
        if native:
            for part in ("fast","protected","background","slow"):
                reps['history_' + part].append(get_state(state, part).numpy())
            # Factual binding available to a diagnostic that is told the requested key.
            reps['history_joint'].append(torch.cat([get_state(state, c) for c in ('fast', 'protected', 'background', 'slow')], 1).numpy())
        preds = []
        final = []
        local = []
        routes = []
        for question in q:
            qs = [question] * len(x)
            z, _, h = runs(m, qs, state)
            preds.append(z[:, -1])
            final.append(h[:, -1])
            if native:
                loc, _, ax = m.parent.local(ids(qs), m.parent.core.emb, state[1])
                local.append(loc[:, -1])
                routes.append(torch.stack([ax[0].argmax(-1), ax[1].argmax(-1)], 2))
        scores.append(torch.stack(preds, 1).reshape(n, 2, 2, -1).numpy())
        reps['final'].append(torch.stack(final, 1).reshape(n * 4, -1).numpy())
        if native:
            reps['input_local'].append(torch.stack(local, 1).reshape(n * 4, -1).numpy())
        if native and start>=len(train):
            lx = ids(x)
            _, _, ax = m.parent.local(lx, m.parent.core.emb, None)
            role = ax[0].argmax(-1)
            addr = ax[1].argmax(-1)
            histroles.append(torch.stack((role, addr), 2).numpy())
            queryroles.append(torch.stack(routes, 1).numpy())
            # Swap each type of carried state independently between the two factual worlds.
            def flip(row):
                if torch.is_tensor(row):
                    return row.reshape(n, 2, *row.shape[1:]).flip(1).reshape_as(row)
                if row is None:
                    return None
                return type(row)((flip(y) for y in row))
            donor = flip(state)
            y = torch.tensor([48 + r[j] for r in group for j in (2, 3)])
            idx = torch.arange(len(x))
            old = preds[0][idx, y] - preds[0][idx, flip(y)]
            dz = flip(preds[0])
            ideal = dz[idx, y] - dz[idx, flip(y)] - old
            transports['full'].extend(ideal.tolist())
            for part in ("fast","protected","background","slow","tail"):
                z, _, _ = runs(m, [q[0]] * len(x), patch(state, donor, part))
                z = z[:, -1]
                transports[part].extend((z[idx, y] - z[idx, flip(y)] - old).tolist())
            if start==len(train):
                direct = m(ids([s + q[0] for s in x]))[:, -1]
                fullerr = float((direct - preds[0]).abs().max())
    split = len(train)
    labels = np.array([[[a, b], [b, a]] for _, _, a, b in train + dev])
    true = labels[split:]
    raw = np.concatenate(scores)[split:]
    p = raw.argmax(-1) - 48
    def rates(p):
        f = p == true
        return dict(single=mean(f), fact_pairs=mean(f.all(axis=1)), query_pairs=mean(f.all(axis=2)), grid=mean(f.all(axis=(1, 2))))
    out = {'dev_worlds': len(dev), 'train_probe_worlds': split, 'head': rates(p), 'probe': {}}
    for name,vals in reps.items():
        vals = np.concatenate(vals)
        if name.startswith("history"):
            px = []
            for j in (0,1):
                y = labels[:, :, j].flatten()
                px.append(probe(vals[:2 * split], vals[2 * split:], y[:2 * split], y[2 * split:]))
            pred = np.stack(px, axis=-1).reshape(-1, 2, 2)
        else:
            y = labels.flatten()
            pred = probe(vals[:4 * split], vals[4 * split:], y[:4 * split], y[4 * split:]).reshape(-1, 2, 2)
        out['probe'][name] = rates(pred)
    if native:
        ar = np.concatenate(histroles)
        qr = np.concatenate(queryroles)
        w = (ar[:, :, 0] == 1) | (ar[:, :, 0] == 3)
        hist = np.bincount(ar[:, :, 1][w], minlength=8)
        p = hist / max(1, hist.sum())
        eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if hist.sum() else 0.0
        out['routing']={"historical_action_counts":np.bincount(ar[:,:,0].ravel(),minlength=4).tolist(),"write_or_reset_by_slot":hist.tolist(),"effective_slots":eff,
        "any_query_role_or_address_switch":mean((qr[:,0]!=qr[:,1]).any(axis=(1,2))),"final_query_address_switch":mean(qr[:,0,-1,1]!=qr[:,1,-1,1]),"query_read_fraction":mean(qr[:,:,:,0]==2)}
        out['abs_margin_shift_after_state_swap'] = {k: mean(np.abs(v)) for k, v in transports.items()}
        out['chunk_parity_max'] = fullerr
    return out

def loadstudy(arg):
    d, arm, scale, seed = arg.split(':')
    seed = int(seed)
    m = build_repair_model(seed) if arm == 'R' else build_model(arm, seed, scale)
    if d!="random":
        if d.endswith(".zip"):
            with zipfile.ZipFile(d) as z:
                raw = z.read('matched_data_weights.pt')
        else:raw=Path(d).read_bytes()
        m.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True))
        h = hashlib.sha256(raw).hexdigest()
    else:h=None
    return (m.eval(), arm, scale, seed, h)

@torch.inference_mode()
def loadtest(m):
    rng = np.random.default_rng(12596)
    n = 384
    data = []
    for i in range(n):
        k = int(rng.integers(4))
        a = int(rng.integers(4))
        b = (a + int(rng.integers(1, 4))) % 4
        others = [x for x in range(4) if x != k]
        rng.shuffle(others)
        v = rng.integers(4, size=3)
        data.append((k, a, b, others, v))
    out = {}
    for mode in ("immediate","short_one_identity","long_one_identity","long_three_identities","target_last","gap_64"):
        cc = []
        diff = []
        predsame = []
        for at in range(0,n,16):
            texts = []
            labels = []
            for k,a,b,others,v in data[at:at+16]:
                num = {'immediate': 0, 'short_one_identity': 1, 'target_last': 3}.get(mode, 3)
                ks = [others[0]] * 3 if mode in ('short_one_identity', 'long_one_identity') else others
                suffix = ''.join((f"!{'abcd'[x]}{y}" for x, y in list(zip(ks, v))[:num]))
                for digit in (a,b):
                    fact = f"!{'abcd'[k]}{digit}"
                    s = suffix + fact if mode == 'target_last' else fact + suffix
                    texts.append(s + ('.' * 64 if mode == 'gap_64' else '') + f"#{'abcd'[k]}00=")
                    labels.append(48 + digit)
            z = m(ids(texts))[:, -1]
            p = z.argmax(-1)
            y = torch.tensor(labels)
            c = (p == y).view(-1, 2)
            cc.extend(c.all(1).tolist())
            predsame.extend((p[::2] != p[1::2]).tolist())
            diff.extend((z[::2, 48:52] - z[1::2, 48:52]).abs().amax(1).tolist())
        out[mode] = {'pairs_correct': sum(cc), 'pairs': n, 'answer_switches': sum(predsame), 'mean_max_digit_logit_change': mean(diff)}
    return out

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    ap.add_argument('--english', action='store_true')
    ap.add_argument('--control', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    m, arm, scale, seed, h = loadstudy(a.checkpoint)
    with torch.inference_mode():
        result = dict(checkpoint_sha256=h, arm=arm, scale=scale, seed=seed, load=loadtest(m))
        for domain in (["symbol","english"] if a.english else ["symbol"]):
            result[domain] = assay(m, arm == 'C' and (not a.control), domain)
    Path(a.out).write_text(json.dumps(result, indent=2))
