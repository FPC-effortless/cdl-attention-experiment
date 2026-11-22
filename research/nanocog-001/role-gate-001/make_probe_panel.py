"""Build a small fourth synthetic factorial panel and log immutable hashes."""
import json,itertools,random
from pathlib import Path
from hashlib import sha256
from cog_comm.role_control import NAMES,COLORS,CELLS,families,world
old=families()
excluded=set(old['fit'])|set(old['control'])
root=Path(__file__).resolve().parent
for n,pin in [('002','ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'),
              ('003','edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84')]:
 raw=(root.parent/f'cog-role-{n}/panel/confirm_judge.json').read_bytes()
 assert sha256(raw).hexdigest()==pin
 new={tuple(map(tuple,r['key'])) for r in json.loads(raw)['families']}
 assert len(new)==120 and not (new&excluded)
 excluded.update(new)
assert len(excluded)==680
universe=set(itertools.product(itertools.combinations(NAMES,2),itertools.combinations(COLORS,2)))
remain=sorted(universe-excluded)
random.Random(20261125).shuffle(remain)
choices=remain[:32]
p=[];j=[]
for key in choices:
 worlds=[world(key,c) for c in CELLS]
 p.extend(" ".join(t) for t in worlds)
 j.append(dict(key=key, answers=[{t[0]:t[2],t[3]:t[5]}[t[-1]] for t in worlds]))
assert len(set(p))==len(p)==256
D=root/'probe_panel';D.mkdir(exist_ok=False)
for n,v in [('actor.json',dict(prompts=p,schema='cog-query-probe.actor.v1')),
            ('judge.json',dict(families=j,schema='cog-query-probe.judge.v1'))]:
 data=(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
 (D/n).write_bytes(data);print(n,sha256(data).hexdigest())
