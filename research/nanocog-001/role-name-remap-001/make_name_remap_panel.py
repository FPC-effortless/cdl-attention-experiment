"""Construct balanced name-remapping strings; no graph/model, predictions or training."""
from collections import Counter
from hashlib import sha256
from itertools import combinations,permutations,product
from pathlib import Path
import json,random
from cog_comm.role_control import NAMES,COLORS,families

ROOT=Path(__file__).resolve().parent
SEED=20261128
FRAGILE=('gio','jay','kai','ola')
PINS=(
 ('cog-role-002/panel/confirm_judge.json','ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'),
 ('cog-role-003/panel/confirm_judge.json','edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84'),
 ('cog-role-diag/probe_panel/judge.json','d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848'),
 ('cog-role-repair-004/repair_panel/judge.json','0a3ae7b6d30ed4554b908c44411cb3e9e74c49e1942ef93c5fe8fad5b5005c23'),
)
NATIVE_SHA='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
ADAPTER_SHA='abfd950297ca1b90846621f770601b7e38081f5c62eea49e4e1c814477674932'
PROOF='9b575f365052d022b2ac4dba046658a2c75e5e2a7ce0d812f811ffb7fd5c8b7e'
def need(x,m):
    if not x:raise ValueError(m)
def h(b):return sha256(b).hexdigest()
def save(p,x):
    blob=(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(blob)
    return h(blob)
def build(dest):
    original=families()
    seen=set(original['fit'])|set(original['control'])
    for rel,pin in PINS:
        content=(ROOT.parent/rel).read_bytes()
        need(h(content)==pin,'wrong prior keys')
        keys={tuple(map(tuple,z['key'])) for z in json.loads(content)['families']}
        need(not seen&keys,'prior key overlap')
        seen|=keys
    need(len(seen)==832,'wrong exclusions')
    train=Counter(n for (ns,_cs) in original['fit'] for n in ns)
    robust=min(permutations([n for n in NAMES if n not in FRAGILE],4),
                 key=lambda r:(sum(abs(train[a]-train[b]) for a,b in zip(FRAGILE,r,strict=True)),r))
    distractors=tuple(n for n in NAMES if n not in set(FRAGILE)|set(robust))
    need(len(distractors)==8,'wrong anchor cohort')
    rng=random.Random(SEED); anchors=list(distractors)*3;rng.shuffle(anchors)
    pairslots=[ij for ij in combinations(range(4),2) for _ in range(4)]
    draws=[]; blocks=[]; fresh=set()
    for i,((j,k),anchor) in enumerate(zip(pairslots,anchors,strict=True)):
        roles=(('F0',FRAGILE[j]),('F1',FRAGILE[k]),('R0',robust[j]),('R1',robust[k]))
        def key(n,colors):return tuple(sorted((n,anchor),key=NAMES.index)),colors
        possible=[c for c in combinations(COLORS,2) if all(key(n,c) not in seen|fresh for _,n in roles)]
        need(bool(possible),'infeasible reserved block')
        colors=rng.choice(possible)
        fresh.update(key(n,colors) for _,n in roles)
        rec=[]
        for assignment,position,asked in product((0,1),repeat=3):
            values=colors if assignment==0 else colors[::-1]
            alternatives={}
            for variant,n in roles:
                foc=(n,'is',values[0]); dist=(anchor,'is',values[1])
                clause=(foc+dist) if position==0 else (dist+foc)
                text=' '.join(clause+(n if asked==0 else anchor,))
                alternatives[variant]=len(draws);draws.append(text)
            rec.append(dict(assignment=assignment,focal_position='first' if position==0 else 'second',
                             queried='focal' if asked==0 else 'distractor',answer=values[asked],foil=values[1-asked],
                             alternatives=alternatives))
        blocks.append(dict(block=i,distractor=anchor,colors=colors,roles=dict(roles),cells=rec))
    need(len(draws)==len(set(draws))==768 and len(fresh)==96 and not fresh&seen,'new identities overlap')
    order=list(range(len(draws)));rng.shuffle(order);inverse={v:i for i,v in enumerate(order)}
    for b in blocks:
        for c in b['cells']:
            c['alternatives']={k:inverse[j] for k,j in c['alternatives'].items()}
    dest.mkdir()
    a=save(dest/'actor.json',dict(schema='cog-role-name-remap.actor.v1',prompts=[draws[j] for j in order]))
    j=save(dest/'judge.json',dict(schema='cog-role-name-remap.judge.v1',panel_seed=SEED,blocks=blocks))
    manifest={'schema':'cog-role-name-remap.manifest.v1','excluded_joint_families':len(seen),'new_joint_families':96,
             'blocks':24,'worlds':768,'focal_class_block_occurrences':dict(Counter(n for b in blocks for n in b['roles'].values())),
             'distractor_occurrences':{n:anchors.count(n) for n in distractors},'fragile_focal':FRAGILE,'robust_focal':robust,
             'focal_train_family_support':{x:train[x] for x in FRAGILE+robust},'actor_sha256':a,'judge_sha256':j,
             'native_backbone_sha256':NATIVE_SHA,'tap1471_adapter_sha256':ADAPTER_SHA,
             'prior_failure_diagnostic_sha256':PROOF,'source_sha256':h(Path(__file__).read_bytes())}
    print('manifest',save(dest/'manifest.json',manifest));print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':build(ROOT/'panel')
