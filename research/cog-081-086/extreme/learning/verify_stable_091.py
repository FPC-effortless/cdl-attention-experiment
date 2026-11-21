"""Check exact stable decisions for original, fresh, budget stress panels."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parent
FILES=('maze-091.json','maze-091-fresh.json','maze-091-budget.json')
OMIT={'elapsed_ns','total_elapsed_ms','total_ms','wall_ms','train_ms'}

def stable(v):
    if isinstance(v,dict):return {k:stable(z) for k,z in v.items() if k not in OMIT}
    if isinstance(v,list):return [stable(z) for z in v]
    return v

def hashes():
    out={}
    for fn in FILES:
        payload=json.loads((ROOT/'results'/fn).read_text())
        body=json.dumps(stable(payload),sort_keys=True,separators=(',',':')).encode()
        out[fn]=hashlib.sha256(body).hexdigest()
    return out

if __name__=='__main__':
    expected=json.loads((ROOT/'STABLE_SHA256.json').read_text())
    seen=hashes()
    assert expected==seen,dict(expected=expected,actual=seen)
    print('Stable decision/evaluation hashes matched:',len(seen))
