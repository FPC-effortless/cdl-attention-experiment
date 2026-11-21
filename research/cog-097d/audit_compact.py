"""Audit GitHub-hosted compact evidence without checkpoints or full raw traces."""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parent
SEEDS=('1701','1707','1711')

def audit(check_sources=True):
    doc=json.loads((ROOT/'GH_RESULTS_COMPACT.json').read_text())
    n=0;verified=0
    for phase in ('d1','d2'):
        assert set(doc[phase])==set(SEEDS)
        for seed in SEEDS:
            row=doc[phase][seed]
            assert row['seed']==int(seed)
            for method,score in row['models'].items():
                ranks=score['per_query_rank']
                assert len(ranks)==score['n']
                assert sum(r==1 for r in ranks)==score['correct']
                assert abs(score['accuracy']-score['correct']/score['n'])<1e-5
                verified+=1
            if check_sources:
                hashes=row.get('source_sha256') or row['source_hashes']
                for mod,want in hashes.items():
                    spec=importlib.util.find_spec(mod)
                    assert spec and spec.origin,mod
                    assert hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()==want,mod
                    n+=1
    return dict(evidence_arms_verified=verified,source_files_verified=n,
                verified_source_bytes=check_sources)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--skip-source-hashes',action='store_true')
    args=p.parse_args();print(json.dumps(audit(not args.skip_source_hashes),indent=2))
