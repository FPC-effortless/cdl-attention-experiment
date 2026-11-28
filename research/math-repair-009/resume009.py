"""Migrate only NumPy float64 development metadata in authenticated stage checkpoints."""
import argparse, hashlib, json, math, shutil
from pathlib import Path
import numpy as np
import torch


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identical(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def normalize(prior, out):
    prior, out = Path(prior), Path(out)
    assert prior.resolve() != out.resolve()
    p = prior / 'checkpoint.pt'
    r = json.loads((prior / 'receipt.json').read_text())
    assert r['terminal_stage'] is True and r['step'] in (2048, 4096, 6144, 8192)
    assert r['targets'] == r['step'] * 16 * 768 and digest(p) == r['sha256']
    assert r['lock']['torch'] == str(torch.__version__) and r['lock']['numpy'] == np.__version__
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(p))
    assert unsafe <= {'numpy._core.multiarray.scalar', 'numpy.dtype'}, sorted(unsafe)
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype('float64'))]):
        cp = torch.load(p, weights_only=True, map_location='cpu')
    assert cp['step'] == r['step'] and cp['lock'] == r['lock'] == json.loads((prior / 'lock.json').read_text())
    assert cp['optimizer']['state'] and cp['weights'] and torch.is_tensor(cp['rng'])
    assert cp['curve'][-1]['step'] == r['step'] and 'dev_bpb' in cp['curve'][-1]
    changes = []
    for i, entry in enumerate(cp['curve']):
        if 'dev_bpb' in entry:
            value = entry['dev_bpb']
            assert type(value) in (float, np.float64) and math.isfinite(value)
            if type(value) is np.float64:
                entry['dev_bpb'] = float(value)
                assert entry['dev_bpb'] == value
                changes.append(i)
    dev = json.loads((prior / 'development.json').read_text())
    assert cp['curve'][-1]['dev_bpb'] == dev['bpb'] and cp['curve'][-1]['dev_targets'] == dev['targets']
    out.mkdir(parents=True, exist_ok=False)
    q = out / 'checkpoint.pt'
    torch.save(cp, q)
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(q)
    reread = torch.load(q, weights_only=True, map_location='cpu')
    assert identical(cp, reread), 'metadata migration changed checkpoint contents'
    for name in ('development.json', 'environment.json', 'lock.json', 'preflight.json'):
        shutil.copyfile(prior / name, out / name)
    audit = {'status': 'SAFE_METADATA_MIGRATION', 'original_checkpoint_sha256': r['sha256'], 'normalized_checkpoint_sha256': digest(q), 'original_run': r['run_id'], 'arm': r['lock']['arm'], 'seed': r['lock']['seed'], 'step': r['step'], 'changed_dev_bpb_entries': changes, 'original_unsafe_globals': sorted(unsafe), 'all_checkpoint_values_and_tensors_preserved': True, 'normalizer_sha256': digest(Path(__file__))}
    r['sha256'] = audit['normalized_checkpoint_sha256']
    r['metadata_migration'] = audit
    (out / 'receipt.json').write_text(json.dumps(r, indent=2))
    (out / 'migration.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prior', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    import recover009
    from data009 import sha
    recover009.corpus(a.data)
    r = json.loads((Path(a.prior) / 'receipt.json').read_text())
    assert str(r['run_id']) == str(a.run_id) and r['step'] == a.stage * 2048
    assert r['lock']['arm'] == a.arm and r['lock']['seed'] == a.seed
    assert r['lock']['source'] == recover009.source(a.arm)
    assert r['lock']['data'] == sha(Path(a.data) / 'manifest.json')
    normalize(a.prior, a.out)
    a.checkpoint = str(Path(a.out) / 'checkpoint.pt')
    recover009.checkpoint(a)
