"""Independent post-score audit of frozen Clause-Order output, no model import/forward.

Do not invoke until all outputs and the official report have been sealed and
the independent score stage is authorized. Opens only constructed study files.
"""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json,math
from pathlib import Path
from struct import pack,unpack

WIDTH=40
ARMS=('sym','order','raw')
SEEDS=(3371,3471,3571)
KEYS=set((e,r) for e in range(24) for r in range(24,28))
VALUES=set(range(28,40))

def require(x,reason):
    if not x: raise ValueError(reason)

def payload(p,h=None):
    require(p.is_file() and not p.is_symlink(),'regular exact path')
    b=p.read_bytes()
    require(h is None or sha256(b).hexdigest()==h,'hash mismatch: '+p.name)
    return b

def read(p,h=None):
    def pairs(a):
        d=dict(a);require(len(d)==len(a),'duplicate JSON keys');return d
    return json.loads(payload(p,h),object_pairs_hook=pairs)

def digest(p):return sha256(payload(p)).hexdigest()

def keypair(x):
    k=tuple(sorted(map(tuple,x)))
    require(len(k)==2 and k[0]!=k[1] and all(z in KEYS for z in k),'exact distinct composite keys')
    return k

def stratum(k):
    a,b=k
    return 0 if a[0]==b[0] else (1 if a[1]==b[1] else 2)

def parse(x):
    require(isinstance(x,list) and len(x)==8 and all(type(k)is int for k in x),'eight literal IDs')
    a,b,q=tuple(x[:2]),tuple(x[3:5]),tuple(x[6:])
    k=keypair((a,b));v=x[2],x[5]
    require(q in k and len(set(v))==2 and all(z in VALUES for z in v),'distinct values and query key')
    return k,int(a!=k[0]),k.index(q),v if a==k[0] else v[::-1],v[0 if q==a else 1]

def panel(xs,yy,j):
    require(len(xs)==len(yy)==64*4*24 and len(set(map(tuple,xs)))==len(xs),'exact held row closure')
    blocks=[]
    for head in range(0,len(xs),24):
        rows=[parse(x) for x in xs[head:head+24]]
        k=rows[0][0];v0=rows[0][3];third={v for m in rows for v in m[3]}-set(v0)
        require(len(third)==1 and stratum(k)==j,'stratum and one genuinely new value')
        x,y=v0; z=third.pop()
        phases=[(x,y),(z,y),(x,z),(y,x),(z,x),(y,z)]
        by={}
        for i,(pair,order,query,vs,correct) in enumerate(rows):
            require(pair==k and correct==yy[head+i] and vs in phases,'exact derived answer and physical state')
            mode=phases.index(vs); item=mode,order,query
            require(item not in by and i==4*mode+2*order+query,'exact producer row wire; independent mapping agrees')
            by[item]=(head+i,correct)
        require(len(by)==24,'complete assignment/edit/order/query panel')
        blocks.append((k,by,phases))
    groups=Counter(k for k,_,_ in blocks)
    require(len(groups)==64 and set(groups.values())=={4},'64 groups each with four histories')
    require(all(len({tuple(sorted(v)) for key,_,ph in blocks if key==k for v in ph})==12 for k in groups),
            'value pairs disjoint across histories per group')
    return blocks

def f32(x):return unpack('<f',pack('<f',x))[0]

def decode(torch,p,h,n):
    b=payload(p,h)
    require(0<len(b)<1_100_000,'bounded output size')
    require(not torch.serialization.get_unsafe_globals_in_checkpoint(BytesIO(b)),'logit wire must have no unsupported globals')
    a=torch.load(BytesIO(b),map_location='cpu',weights_only=True)
    require(type(a)is torch.Tensor and a.dtype==torch.float32 and a.device.type=='cpu' and a.is_contiguous()
            and tuple(a.shape)==(n,WIDTH) and bool(torch.isfinite(a).all()),'finite complete float tensor')
    return a.tolist()

def eval_panel(blocks,z):
    top=[]; chosen=[]
    for v in z:
        hi=max(v);first=v.index(hi);top.append(first if v.count(hi)==1 else None);chosen.append(first)
    counts=Counter({x:0 for x in ('individual','ordinary','first','second','both_order_triplets','swap_pairs',
        'joint','order_agreement','wrong_agreement','non_tied_order_agreement','non_tied_wrong_agreement',
        'direct_first_triplets','direct_second_triplets','direct_fixed_query_both_order_triplets','direct_opposite_query_order_quartets',
        'one_order_only','absolute_tied_rows')})
    widest=0.0
    for _,b,_ in blocks:
        good={v:top[i]==gold for v,(i,gold) in b.items()}
        counts['individual']+=sum(good.values())
        counts['absolute_tied_rows']+=sum(top[i] is None for i,_ in b.values())
        counts['ordinary']+=all(good[m,o,q] for m in (0,3) for o in (0,1) for q in (0,1))
        trips={(base,o,q):all(good[m,o,q] for m in range(base,base+3)) for base in (0,3) for o in (0,1) for q in (0,1)}
        counts['first']+=all(trips[base,o,o] for base in (0,3) for o in (0,1))
        counts['second']+=all(trips[base,o,1-o] for base in (0,3) for o in (0,1))
        counts['both_order_triplets']+=all(trips.values())
        counts['joint']+=all(good.values())
        counts['swap_pairs']+=sum(good[0,o,q] and good[3,o,q] for o in (0,1) for q in (0,1))
        counts['direct_first_triplets']+=sum(trips[base,o,o] for base in (0,3) for o in (0,1))
        counts['direct_second_triplets']+=sum(trips[base,o,1-o] for base in (0,3) for o in (0,1))
        counts['direct_fixed_query_both_order_triplets']+=sum(trips[base,0,q] and trips[base,1,q] for base in (0,3) for q in (0,1))
        counts['direct_opposite_query_order_quartets']+=sum(all(good[m,o,q] for o in (0,1) for q in (0,1)) for m in range(6))
        for m in range(6):
            for q in (0,1):
                i,g=b[m,0,q];j,h=b[m,1,q]
                require(g==h,'same key and gold under order move')
                a,d=good[m,0,q],good[m,1,q]
                counts['one_order_only']+=a!=d
                same=chosen[i]==chosen[j]; strict=top[i] is not None and top[i]==top[j]
                counts['order_agreement']+=same
                counts['wrong_agreement']+=same and not a and not d
                counts['non_tied_order_agreement']+=strict
                counts['non_tied_wrong_agreement']+=strict and not a and not d
                widest=max(widest,max(abs(f32(x-y)) for x,y in zip(z[i],z[j],strict=True)))
    return dict(counts,histories=len(blocks),max_order_logit_delta=widest)

def held_pass(s):
    n=s['histories']
    return (100*s['individual']>=95*24*n and 100*s['ordinary']>=90*n
        and all(100*s[q]>=90*n for q in ('first','second','both_order_triplets'))
        and 100*s['swap_pairs']>=90*4*n)

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','results','run-source','out'):ap.add_argument('--'+k,type=Path,required=True)
    for k in ('custody-sha','source-sha','result-sha'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args()
    proof=read(a.data/'custody.json',a.custody_sha);src=payload(a.run_source,a.source_sha)
    report=read(a.results/'RESULTS.json',a.result_sha)
    names={'train.json'}|{f'held-{j}-{kind}.json' for j in range(3) for kind in ('prompts','truth')}
    require(set(proof)=={'files','fit_key_pairs','held_key_pairs'} and set(proof['files'])==names
            and proof['fit_key_pairs']==[80,1040,3248] and proof['held_key_pairs']==[64]*3,'exact frozen custody metadata')
    keys={f'{x}-{y}' for x in ARMS for y in SEEDS};indexed={}
    require(len(report)==9,'exact report roster')
    for r in report:
        k=f"{r['arm']}-{r['seed']}";require(k in keys-indexed.keys(),'no missing/duplicated arm seed');indexed[k]=r
    t=read(a.data/'train.json',proof['files']['train.json'])
    train=[{keypair(g) for g in pool} for pool in t['train']]
    panels=[]
    for j,pool in enumerate(train):
        require(len(pool)==proof['fit_key_pairs'][j] and all(stratum(g)==j for g in pool),'train key inventory')
        for g,vs in t['cal'][j]:require(keypair(g) in pool and len(vs)==2,'TRAIN-only adequacy view')
        pp=f'held-{j}-prompts.json';yy=f'held-{j}-truth.json'
        blocks=panel(read(a.data/pp,proof['files'][pp]),read(a.data/yy,proof['files'][yy]),j)
        require(not pool&{b[0] for b in blocks},'held key pair excludes TRAIN')
        panels.append(blocks)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases={}; original={'histories','individual','ordinary','first','second','both_order_triplets','swap_pairs','joint',
                         'order_agreement','wrong_agreement','max_order_logit_delta'}
    for name in sorted(keys):
        f=a.results/name;z=read(f/'fit.json');p=read(f/'predict.json')
        arm,seed=name.split('-')
        require(z['source_sha']==a.source_sha and z['data_sha']==proof['files']['train.json'] and len(p)==3
                and z['arm']==arm and z['seed']==int(seed) and z['steps']==1024
                and z['parameters']==(35944 if arm=='raw' else 37600)
                and math.isfinite(z['cpu_fit_seconds']) and z['cpu_fit_seconds']>0
                and len(z['losses'])==1024 and all(math.isfinite(y) and y>=0 for y in z['losses'])
                and digest(f/'weights.pt')==z['weight_sha'],'fit/predict settings and weight bytes; never decoded')
        require({k:v for k,v in z.items() if k!='losses'}=={k:v for k,v in indexed[name].items() if k not in ('held','passed')},
                'official source metadata matches retained FIT record')
        scopes=[]
        for j,b in enumerate(panels):
            v=p[j];n=f'held-{j}-logits.pt'
            require(set(v)=={'prompt_sha','logit_sha'} and v['prompt_sha']==proof['files'][f'held-{j}-prompts.json'],'exact actor prompt order')
            s=eval_panel(b,decode(torch,f/n,v['logit_sha'],6144));q=indexed[name]['held'][j]
            require(all(s[k]==q[k] for k in original),'stored scorer vs independent recount mismatch')
            scopes.append(s)
        ab=all(held_pass(s) for s in scopes)
        cal=z['cal']; ca=all(100*x['individual']>=95*24*x['histories'] and 100*x['ordinary']>=90*x['histories'] for x in cal)
        require(ca==z['adequate']==indexed[name]['adequate'] and (ca and ab)==indexed[name]['passed'],'separate train/held gate agreement')
        cases[name]=dict(held=scopes,train_adequate=ca,held_absolute=ab,qualified=ca and ab,
            fit_sha256=digest(f/'fit.json'),prediction_sha256=digest(f/'predict.json'))
    out=dict(status='INDEPENDENT_CLAUSE_ORDER_RECOUNT',source_sha256=digest(Path(__file__)),model_source_sha256=sha256(src).hexdigest(),
        custody_sha256=a.custody_sha,result_sha256=a.result_sha,cases=cases,
        direct_denominators=dict(physical_first_triplets=1024,physical_second_triplets=1024,
            fixed_query_both_order_triplets=1024,opposite_query_order_quartets=1536,order_pairs=3072),
        note='History-level both_order_triplets equals joint24 by definition. Torch only deserialized retained logit tensors.',
        model_calls=0,weights_loaded=False,untouched_other_ranks=True)
    with a.out.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
    print(digest(a.out))
if __name__=='__main__':main()
