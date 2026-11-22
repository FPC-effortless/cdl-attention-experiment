"""Build a new prospective synthetic factorial panel; no weights or outcomes."""
import itertools,json,random
from pathlib import Path
from hashlib import sha256
from cog_comm.role_control import CELLS,COLORS,NAMES,families,world
R=Path(__file__).resolve().parent
records=[
 ('cog-role-002/panel/confirm_judge.json','ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'),
 ('cog-role-003/panel/confirm_judge.json','edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84'),
 ('cog-role-diag/probe_panel/judge.json','d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848'),
]
v=families()
familiar_names={k[0] for k in v['fit']}
seen=set(v['control'])|set(v['fit'])
assert len(seen)==440
for path,pin in records:
 b=(R.parent/path).read_bytes()
 assert sha256(b).hexdigest()==pin
 f={tuple(map(tuple,z['key'])) for z in json.loads(b)['families']}
 assert not seen&f
 seen.update(f)
assert len(seen)==712
keys=sorted(set(itertools.product(itertools.combinations(NAMES,2),itertools.combinations(COLORS,2)))-seen)
random.Random(20261127).shuffle(keys)
chosen=keys[:120];assert len(set(chosen))==120
actor=[];truth=[]
for i,key in enumerate(chosen):
 ps=[world(key,c) for c in CELLS];actor.extend(' '.join(p) for p in ps)
 a=[{p[0]:p[2],p[3]:p[5]}[p[-1]] for p in ps]
 truth.append(dict(key=key,rows=list(range(8*i,8*i+8)),answers=a,
                   familiar_name_pair=key[0] in familiar_names))
assert len(actor)==len(set(actor))==960
out=R/'repair_panel';out.mkdir(exist_ok=False)
manifest={}
for n,d in [('actor.json',dict(schema='cog-comm-role-001.actor.v1',prompts=actor)),
            ('judge.json',dict(panel='repair-004-fresh',families=truth))]:
 b=(json.dumps(d,separators=(',',':'),sort_keys=True)+'\n').encode()
 (out/n).write_bytes(b);manifest[n]=sha256(b).hexdigest()
manifest['excluded_families']=712;manifest['novel_name_pair_families']=sum(not x['familiar_name_pair'] for x in truth)
(out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
print(json.dumps(manifest,sort_keys=True))
