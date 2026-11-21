"""Content-bind dataset revisions, split members and installed navigation source."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import gymnasium
import minigrid

from tasks import inventory, MULTI, QUIX


def hashes(root, paths):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def digest(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build(data):
    out = {}
    splits = {s:inventory(data,s)[0] for s in ('train','test')}
    for d in ('software','dialogue','interactive'):
        ids = {s:[v for k,v in rows if k==d] for s,rows in splits.items()}
        if set(map(str,ids['train'])) & set(map(str,ids['test'])):
            raise ValueError('train/test identity overlap')
        if d=='software':
            root = data/'quix'
            files = [root/'python_testcases/load_testdata.py']
            for name in ids['train'] + ids['test']:
                files += [root/'python_programs'/(name+'.py'),root/'python_testcases'/('test_'+name+'.py'),
                          root/'json_testcases'/(name+'.json')]
            src = dict(repository='jkoppel/QuixBugs',commit=QUIX,files=hashes(root,files))
        elif d=='dialogue':
            root = data/'mw'
            src = dict(repository='budzianowski/multiwoz',commit=MULTI,
                       files=hashes(root,[root/'data/MultiWOZ_2.2'/s/'dialogues_001.json' for s in ids]))
        else:
            src = {}
            for package in (gymnasium,minigrid):
                root = Path(package.__file__).parent
                src[package.__name__] = dict(version=package.__version__,python_tree=digest(hashes(root,root.rglob('*.py'))))
        out[d] = dict(source=src,source_sha256=digest(src),split_members=ids,split_sha256=digest(ids))
    return dict(experiment='COG-ATTN-099',role='public internal development panel',benchmarks=out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data',type=Path)
    p.add_argument('out',type=Path)
    cfg=p.parse_args()
    cfg.out.write_text(json.dumps(build(cfg.data.resolve()),indent=2)+'\n')
