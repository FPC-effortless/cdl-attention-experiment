"""Build fresh label-free actor and separate semantics; no model/fit."""
from collections import defaultdict,Counter
from hashlib import sha256
from itertools import combinations,product
from pathlib import Path
from random import Random
import json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cog-role-name-consist-001'))
from make_fresh_primary import prior_keys,COLORS,NAMES
ROOT=HERE.parent/'cog-role-name-consist-001/eval_panels'
O='4b36c29d4dc1966163bddffde98d4e3e88b504455630c66532833c06cbb01459'
C='e21a40fe1c67071caa6981ca646349b3829847495478a338dd57026b37e550f7'
def require(x,m):
    if not x:raise ValueError(m)
def sha(b):return sha256(b).hexdigest()
def read(p,pin):
    b=p.read_bytes();require(sha(b)==pin,'input byte closure');return json.loads(b)
def save(p,x):
    b=(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(b)
    return sha(b)
def build(out):
    rng=Random(20261201)
    prev=prior_keys()
    a=read(ROOT/'ordinary_judge.json',O)['families'];b=read(ROOT/'contrast_judge.json',C)['blocks']
    seen1={tuple(map(tuple,f['key'])) for f in a}; seen2={(tuple(sorted((z,d['distractor']))),tuple(d['colors']))
            for d in b for z in d['roles'].values()}
    require(len(prev)==928 and len(seen1)==120 and len(seen2)==96 and not (prev&seen1 or prev&seen2 or seen1&seen2),'prior disjoint')
    prior=prev|seen1|seen2
    candidates=defaultdict(list)
    for np,cp in product(combinations(NAMES,2),combinations(COLORS,2)):
        if (np,cp) not in prior:candidates[np].append(cp)
    available=sorted(p for p,c in candidates.items() if len(c)>=4)
    require(len(available)>=48,'too few name-pair splits')
    for n in range(4096):
        draw=rng.sample(available,48)
        fit,held=draw[:32],draw[32:]
        if all(set(n for pair in q for n in pair)==set(NAMES) for q in (fit,held)):break
    else:raise ValueError('no deterministic fully supported cohort')
    groups={'probe_fit':fit,'probe_held':held}
    texts=[];truth=[];keyset=set()
    for split,pairs in groups.items():
        for np in pairs:
            for color in rng.sample(candidates[np],4):
                require((np,color) not in prior|keyset,'fresh joint key')
                keyset.add((np,color));families=[]
                for s,o,q in product((0,1),repeat=3):
                    assigned=color if s==0 else color[::-1]
                    cl=tuple((name,'is',v) for name,v in zip(np,assigned,strict=True))
                    if o:cl=cl[::-1]
                    row=len(texts);text=' '.join(cl[0]+cl[1]+(np[q],))
                    require(assigned[q]==color[s^q],'binding bit construction')
                    families.append(dict(row=row,assignment=s,rendered_order=o,query_index=q,query_name=np[q],
                      binding_bit=s^q,answer=assigned[q],foil=assigned[1-q],query_in_rendered_first=int(q==o)))
                    texts.append(text)
                truth.append(dict(ordinal=len(truth),split=split,name_pair=np,colors=color,cells=families))
    require(len(keyset)==len(truth)==192 and len(texts)==len(set(texts))==1536,'cohort integrity')
    require(all({c['answer'] for fa in truth if fa['split']==sp for c in fa['cells']}==set(COLORS) for sp in groups),'answer classes')
    order=list(range(len(texts)));rng.shuffle(order);ref={v:i for i,v in enumerate(order)}
    for fa in truth:
        for z in fa['cells']:z['row']=ref[z['row']]
    out.mkdir()
    pa=save(out/'actor.json',dict(schema='cog-info-001.actor.v1',prompts=[texts[i] for i in order]))
    pj=save(out/'judging_contract.json',dict(schema='cog-info-001.ridge-panel.v1',families=truth))
    stats={}
    for sp,pairs in groups.items():
        have=Counter(x for p in pairs for x in p)
        stats[sp]=dict(name_pairs=len(pairs),families=len(pairs)*4,worlds=len(pairs)*32,
            lexical_focal_worlds={n:v*16 for n,v in have.items()})
    m=save(out/'manifest.json',dict(schema='cog-info-001.manifest.v1',seed=20261201,actor_sha256=pa,
         judge_sha256=pj,prior_family_count=len(prior),disjoint_name_pair_groups=True,
         generation_attempts=n+1,coverage=stats))
    print(pa,pj,m)
if __name__=='__main__':build(HERE/'trace_panel')
