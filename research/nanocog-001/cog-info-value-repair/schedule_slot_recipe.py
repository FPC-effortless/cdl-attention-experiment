"""Index-only fixed 1280-update schedule; no TRAIN or held member sampled."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
import json
SEEDS=(2071,2171,2271)
ENDPOINTS=(128,1280)
R=Path(__file__).resolve().parent
FMT='info002.value-repair.abstract-slot-schedule.v1'
def need(ok,m):
    if not ok:raise ValueError(m)
def repeated(n,rounds,rng):
    out=[]
    for _ in range(rounds):
        ids=list(range(n));rng.shuffle(ids);out.extend(ids)
    return out
def make(seed):
    need(seed in SEEDS,'unregistered paired seed')
    tr=repeated(256,5,Random(seed^0xc0612001))
    ordinary=repeated(640,10,Random(seed^0xc0612002))
    order=Random(seed^0xc0612003)
    out=[];u=0
    for k in range(640):
        mixedfirst=bool(order.getrandbits(1))
        for mixed in (mixedfirst,not mixedfirst):
            n=2 if mixed else 8
            row=dict(ordinary_row_slots=ordinary[u:u+n],edit_triple_slots=tr[2*k:2*k+2] if mixed else [])
            out.append(row);u+=n
    need(u==6400 and len(out)==1280,'total schedule scope')
    need(Counter(j for row in out for j in row['ordinary_row_slots'])==Counter({i:10 for i in range(640)})
      and Counter(j for row in out for j in row['edit_triple_slots'])==Counter({i:5 for i in range(256)})
      and all(len(row['ordinary_row_slots'])+3*len(row['edit_triple_slots'])==8 and
          len(set(row['edit_triple_slots']))==len(row['edit_triple_slots']) for row in out),'exact budget geometry')
    meta={}
    for e in ENDPOINTS:
        m=sum(bool(r['edit_triple_slots']) for r in out[:e]); t=sum(len(r['edit_triple_slots']) for r in out[:e])
        n=sum(len(r['ordinary_row_slots']) for r in out[:e])
        need((m,t,n)==(e//2,e,5*e),'matched endpoint exposure')
        meta[str(e)]=dict(mixed_batches=m,ordinary_only_batches=e-m,
             edit_triple_presentations=t,edit_rows=3*t,ordinary_rows=n,total_world_rows=3*t+n)
    return dict(schema=FMT,seed=seed,endpoint_counts=meta,steps=out)
def write(p,x):
    b=(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    with p.open('xb') as f:f.write(b)
    return sha256(b).hexdigest()
if __name__=='__main__':
    p=R/'abstract_schedule';p.mkdir()
    r={}
    for seed in SEEDS:
        x=make(seed);r[str(seed)]=dict(schedule_sha256=write(p/f'slots-{seed}.json',x),counts=x['endpoint_counts'])
    print(write(p/'manifest.json',dict(schema=FMT,head_scope='matched arms one fixed weak backbone',
     seed_roster=SEEDS,actual_train_or_held_rows_sampled=False,ordinary_slot_roster=640,edit_triple_slot_roster=256,
     schedules=r)))
