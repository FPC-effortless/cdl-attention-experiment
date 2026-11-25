"""Prescheduled, slot-only training exposure, with no data-member realization."""
from collections import Counter
from hashlib import sha256
from random import Random
import json
from joint_local_objective import SEEDS,ENDPOINTS,need

def encoded(x):return (json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()

def create(seed):
    need(seed in SEEDS,'unregistered fresh seed')
    def repeats(count,periods,tag):
        r=Random(seed^tag);out=[]
        for _ in range(periods):
            items=list(range(count));r.shuffle(items);out.extend(items)
        return out
    e=repeats(128,5,0x4a001001);o=repeats(640,10,0x4a001002);rng=Random(seed^0x4a001003)
    p=0;s=[]
    for k in range(640):
        first=bool(rng.getrandbits(1))
        for mixed in (first,not first):
            n=2 if mixed else 8
            s.append(dict(bundle_slot=e[k] if mixed else None,ordinary_slots=o[p:p+n]));p+=n
    need(Counter(q['bundle_slot'] for q in s if q['bundle_slot'] is not None)==Counter({i:5 for i in range(128)})
         and Counter(v for q in s for v in q['ordinary_slots'])==Counter({i:10 for i in range(640)})
         and len(s)==1280 and p==6400,'registered exposures')
    return dict(schema='cog-joint-local-001.slot-schedule.v1',seed=seed,steps=s)

def signature(seed):return sha256(encoded(create(seed))).hexdigest()

def manifest():
    return dict(schema='cog-joint-local-001.abstract-schedule-manifest.v1',role='SLOT_ONLY',
       seeds=list(SEEDS),endpoint_updates=list(ENDPOINTS),history_slot_roster=128,ordinary_row_slot_roster=640,
       schedules={str(s):dict(file=f'slots-{s}.json',sha256=signature(s)) for s in SEEDS})

def write_abstract(root):
    root.mkdir(mode=0o700)
    for seed in SEEDS:
        with (root/f'slots-{seed}.json').open('xb') as f:f.write(encoded(create(seed)))
    with (root/'manifest.json').open('xb') as f:f.write(encoded(manifest()))

# A caller with explicit capacity/data-stage authorization can write only
# index schedules via write_abstract; this module has no CLI and no side effects.
