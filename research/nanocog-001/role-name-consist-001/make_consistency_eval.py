"""Preregistered synthetic capability and remap panels; no outcomes or weights."""
from collections import Counter
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import json,random
from cog_comm.role_control import NAMES,COLORS,CELLS,families,world
ROOT=Path(__file__).resolve().parent
PINS=[
 ('cog-role-002/panel/confirm_judge.json','ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'),
 ('cog-role-003/panel/confirm_judge.json','edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84'),
 ('cog-role-diag/probe_panel/judge.json','d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848'),
 ('cog-role-repair-004/repair_panel/judge.json','0a3ae7b6d30ed4554b908c44411cb3e9e74c49e1942ef93c5fe8fad5b5005c23'),
 ('cog-role-name-remap-001/panel/judge.json','3e1eecfc644f11bad7c722cce6280a39f2650fe93d750a1a9fb52a0bc7d42947'),
]
FRAG=('gio','jay','kai','ola');ROB=('di','pam','eli','lou')
def pin(path,expected):
 b=path.read_bytes();assert sha256(b).hexdigest()==expected
 return json.loads(b)
def emit(out,name,x):
 b=(json.dumps(x,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode()
 with (out/name).open('xb') as f:f.write(b)
 return sha256(b).hexdigest()
def main():
 o=families();e=set(o['fit'])|set(o['control'])
 for f,h in PINS:
  v=pin(ROOT.parent/f,h)
  if 'families' in v:
   n={tuple(map(tuple,x['key'])) for x in v['families']}
  else:
   n={(tuple(sorted([name,b['distractor']])),tuple(b['colors']))
      for b in v['blocks'] for name in b['roles'].values()}
  assert not e&n
  e.update(n)
 assert len(e)==928
 train_names={k[0] for k in o['fit']}
 available=sorted(set(product(combinations(NAMES,2),combinations(COLORS,2)))-e)
 rng=random.Random(20261129);rng.shuffle(available)
 keys=available[:120];e.update(keys)
 ap=[];js=[]
 for i,case in enumerate(keys):
  p=[world(case,c) for c in CELLS]
  ap.extend(' '.join(q) for q in p)
  js.append(dict(key=case,rows=list(range(8*i,8*i+8)),
    answers=[{z[0]:z[2],z[3]:z[5]}[z[-1]] for z in p],
    familiar_name_pair=case[0] in train_names))
 seed=20261130; rng=random.Random(seed)
 distractors=[n for n in NAMES if n not in FRAG+ROB]*3;rng.shuffle(distractors)
 pairs=[c for c in combinations(range(4),2) for _ in range(4)]
 secondary=[];rows=[]
 for n,((i,j),dist) in enumerate(zip(pairs,distractors,strict=True)):
  roles=dict(F0=FRAG[i],F1=FRAG[j],R0=ROB[i],R1=ROB[j])
  def key(x,c):return tuple(sorted((x,dist))),c
  options=[x for x in combinations(COLORS,2) if all(key(v,x) not in e for v in roles.values())]
  assert options
  c=rng.choice(options);e.update(key(v,c) for v in roles.values())
  cells=[]
  for a,order,q in product((0,1),repeat=3):
   fv,dv=c if a==0 else c[::-1];dest={}
   for k,v in roles.items():
    statement=(v,'is',fv);d=(dist,'is',dv)
    text=(statement+d if order==0 else d+statement)+(v if q==0 else dist,)
    dest[k]=len(rows);rows.append(' '.join(text))
   cells.append(dict(assignment=a,focal_position='first' if order==0 else 'second',
    queried='focal' if q==0 else 'distractor',answer=fv if q==0 else dv,foil=dv if q==0 else fv,alternatives=dest))
  secondary.append(dict(block=n,distractor=dist,colors=c,roles=roles,cells=cells))
 ix=list(range(768));rng.shuffle(ix);inv={j:i for i,j in enumerate(ix)}
 for b in secondary:
  for c in b['cells']:c['alternatives']={k:inv[v] for k,v in c['alternatives'].items()}
 out=ROOT/'eval_panels';out.mkdir(exist_ok=False)
 assert len(set(ap))==960 and len(set(rows))==768 and not set(ap)&set(rows) and len(e)==1144
 p={}
 p['ordinary_actor.json']=emit(out,'ordinary_actor.json',dict(schema='cog-comm-role-001.actor.v1',prompts=ap))
 p['ordinary_judge.json']=emit(out,'ordinary_judge.json',dict(panel='name-consistency-fresh',families=js))
 p['contrast_actor.json']=emit(out,'contrast_actor.json',dict(schema='cog-role-name-remap.actor.v1',prompts=[rows[i] for i in ix]))
 p['contrast_judge.json']=emit(out,'contrast_judge.json',dict(schema='cog-role-name-remap.judge.v1',panel_seed=seed,blocks=secondary))
 p['ordinary_unseen_name_pairs']=sum(not f['familiar_name_pair'] for f in js)
 p['prior_disjoint_joint_families']=928
 p['ordinary_families']=120;p['contrast_families']=96;p['contrast_blocks']=24
 emit(out,'manifest.json',p);print(json.dumps(p,sort_keys=True))
if __name__=='__main__':main()
