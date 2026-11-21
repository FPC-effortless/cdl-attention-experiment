"""Verify non-timing outcomes against frozen SHA-256 receipt.

CPU times are retained in raw results, excluded only from stable digests.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAMES=('maze-092.json','sliding-093.json','sliding-093b.json','sliding-093c.json',
       'sudoku-094.json','chess-095.json')
EXCLUDE={'elapsed_ns','policy_ms','total_ms_median','train_ms','elapsed_ms','wall_ms',
         'timings','train_ns','total_elapsed_ms'}

def stable(obj):
    if isinstance(obj,dict):return {k:stable(v) for k,v in obj.items() if k not in EXCLUDE}
    if isinstance(obj,list):return [stable(v) for v in obj]
    return obj

def sha(obj):return hashlib.sha256(json.dumps(stable(obj),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def run(write=False):
    records={n:sha(json.loads((ROOT/'results'/n).read_text())) for n in NAMES}
    artifact=ROOT/'STABLE_SHA256.json'
    if write:artifact.write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
    else:
        expected=json.loads(artifact.read_text())
        if records!=expected:
            for n in NAMES:
                if records[n]!=expected.get(n):print('MISMATCH:',n,records[n],expected.get(n))
            raise AssertionError('stable deterministic evidence does not match frozen manifest')
    print(json.dumps(records,indent=2,sort_keys=True))
    return records
if __name__=='__main__':run('--write' in sys.argv)
