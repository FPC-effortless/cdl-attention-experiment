"""Aggregate traces without counting repetitions of one fact as independent evidence."""
import argparse
from collections import defaultdict
from statistics import median
from common import Path, np, sha, read, save
from fit import CONFIGS


def probe(rows):
    out={}
    for mode in ('full','masked','no_lookup'):
        pairs=defaultdict(list)
        for row in rows:
            pairs[(row['family'],row['who'])].append(row)
        assert len(pairs)==8 and all({x['flip'] for x in v}=={0,1} for v in pairs.values())
        out[mode]=dict(top=sum(r[mode]['top']==r['gold'] for r in rows),preferred=sum(r[mode]['odds']>0 for r in rows),
            both=sum(all(r[mode]['odds']>0 for r in pair) for pair in pairs.values()),
            both_top=sum(all(r[mode]['top']==r['gold'] for r in pair) for pair in pairs.values()))
    out.update(n=len(rows),median_ratio=median(x['ratio'] for x in rows),changed_top=sum(
        r['full']['top']!=r['masked']['top'] for r in rows))
    return out


def dialogue(rows):
    switched=[r['before_old']>r['before_new'] and r['after_new']>r['after_old'] for r in rows]
    return dict(n=len(rows),dialogs=len({r['dialogue'] for r in rows}),switched=sum(switched),
        after=sum(r['after_new']>r['after_old'] for r in rows),exact=sum(r['after_new_exact'] for r in rows),
        recent=sum(not r['before_literal'] and r['after_literal'] for r in rows))


def difference(rows, other):
    groups=defaultdict(list)
    for a,b in zip(rows,other,strict=True):
        assert (a['dialogue'],a['turn'],a['slot'])==(b['dialogue'],b['turn'],b['slot'])
        def correct(v):
            return int(v['before_old']>v['before_new'] and v['after_new']>v['after_old'])
        groups[a['dialogue']].append(correct(a)-correct(b))
    score=np.array([[sum(g),len(g)] for g in groups.values()])
    rng=np.random.default_rng(431)
    v=score[rng.integers(len(score),size=(5000,len(score)))].sum(1)
    return dict(count=int(score[:,0].sum()),ci95_rate=np.quantile(v[:,0]/v[:,1],[.025,.975]).tolist())


def aggregate(a):
    result=dict(fits={},probes={},text={},dialogue={},paired={},timing={},training={})
    for n in CONFIGS:
        f=read(a.out/(n+'.fit.json'))
        assert f['head_sha']==sha(a.out/(n+'.pt'))
        result['fits'][n]={k:v for k,v in f.items() if k not in ('curve','grad_norm')}
        rows=read(a.out/(n+'.probes.json'))
        result['probes'][n]={f'{k}/{d}':probe([r for r in rows if r['kind']==k and r['delay']==d])
            for k in ('literal','reword') for d in (256,1024,4096,16384)}
        nat=read(a.out/(n+'.text.json'))
        result['text'][n]={k:sum(v['nll'] for v in rr)/sum(v['count'] for v in rr) for k,rr in nat.items()}
    for split in ('known','fresh'):
        recorded={}
        for file in sorted(a.out.glob('*.'+split+'.json')):
            name=file.name.split('.')[0]
            recorded[name]=read(file)
        result['dialogue'][split]={n:dialogue(v) for n,v in recorded.items()}
        for name in ('route_cog_control','route_mamba_control','route_revision_off','parent_cog_revision'):
            result['paired'][split+'/'+name]=difference(recorded['route_revision'],recorded[name])
    timing=read(a.out/'timing.json')
    for name in sorted({r['name'] for r in timing}):
        rr=[r for r in timing if r['name']==name]
        result['timing'][name]=dict(decode_tps=32/median(x['decode_s'] for x in rr),
            request_ms=1000*median(x['request_s'] for x in rr),prefill_encode_ms=1000*median(x['encode_prefill_s'] for x in rr))
    train=read(a.out/'train_replay.json')
    for name in sorted({r['name'] for r in train}):
        rr=[r for r in train if r['name']==name and r['kind']=='reword']
        pairs=defaultdict(list)
        for r in rr:pairs[r['family'],r['delay'],r['who']].append(r)
        result['training'][name]=dict(n=len(rr),exact=sum(r['top']==r['gold'] for r in rr),
            both=sum(all(r['odds']>0 for r in pair) for pair in pairs.values()),pairs=len(pairs))
    prior=read(a.out/'attribution_002.json')
    result['attribution_002']={str(d):{k:sum(r[k]==r['gold'] for r in prior if r['delay']==d) for k in
        ('native','without_lookup','table')} for d in (256,1024)}
    result['usage']=read(a.out/'usage.json')
    result['gradient']={}
    for row in read(a.out/'gradient.json'):
        result['gradient'].setdefault(row['name']+('/short' if row['short'] else '/long'),[]).append(row['prefix_feature_gradient'])
    result['gradient']={k:median(v) for k,v in result['gradient'].items()}
    save(a.out/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    aggregate(p.parse_args())
