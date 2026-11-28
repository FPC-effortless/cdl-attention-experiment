"""Extend exact #119 WholeCOG constructor without modifying its source files."""
from pathlib import Path
import sys, hashlib, json, itertools
BASE=Path(__file__).resolve().parent.parent/'cog_unified_search_002'
if not BASE.is_dir(): BASE=Path(__file__).resolve().parent.parent/'unified-search-002'
LOCK={'integrated.py':'e4dd28e559ea039c7ecf4634013d4ffb7becd955','variants.py':'76ece3b4c83a30e282141db7a29beecb10a5d41e','model.py':'31c3555521469425d884c700aed01b48c4fab50e','knowledge.py':'c51c1613a76cc110eb08a72543b32eda01eaa5cc','train.py':'decb4b15c5a35f3e4bc64a50d94b0a21e863e5e5'}
def blob(p):
    b=Path(p).read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
assert all(blob(BASE/k)==v for k,v in LOCK.items())
sys.path.insert(0,str(BASE))
import integrated as I
from variants import ARMS as PATTERNS
BASE_ARMS=dict(I.ARMS)
PRIMARY={f'{p}_{b}_{a}':(p,b,a,'random_full') for p,b,a in itertools.product(PATTERNS,('native','presence'),('native','factor'))}
assert len(PRIMARY)==64 and all(PRIMARY[a]==v for a,v in BASE_ARMS.items() if a in PRIMARY)
I.ARMS.update(PRIMARY)
assert len(I.ARMS)==69
WholeCOG=I.WholeCOG
if __name__=='__main__':
 print(json.dumps({'frozen_parent':BASE_ARMS,'expanded':I.ARMS,'new_primary':sorted(set(PRIMARY)-set(BASE_ARMS))},indent=2))
