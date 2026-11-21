"""Construct public tables and paired article-bootstrap intervals from immutable receipts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def doc_vector(record,key,mixture=False):
    bydoc=defaultdict(lambda:[0.,0])
    for r in record['rows'][key]:
        val=r['mixture_nll'][0] if mixture else r['nll']
        row=bydoc[r['doc']]
        row[0]+=val
        row[1]+=r['count']
    return np.array([bydoc[d] for d in sorted(bydoc)])


def paired(records,group,seeds,left,right,split,cache):
    matrices=[]
    for s in seeds:
        suffix='-continued' if group=='extended' else ''
        tag='scale' if suffix else group
        a=records[f'{tag}-{left}-{s}{suffix}']
        b=records[f'{tag}-{right}-{s}{suffix}']
        key=split+'/full'
        assert [(r['doc'],r['offset'],r['count']) for r in a['rows'][key]]==[(r['doc'],r['offset'],r['count']) for r in b['rows'][key]]
        v,w=doc_vector(a,key),doc_vector(b,key,cache)
        matrices.append(np.column_stack([v[:,0]-w[:,0],v[:,1]]))
    a=np.mean(matrices,0)
    rng=np.random.default_rng(9201)
    draws=a[rng.integers(len(a),size=(5000,len(a)))].sum(1)
    data=draws[:,0]/draws[:,1]
    return dict(delta=float(a[:,0].sum()/a[:,1].sum()),interval95=np.quantile(data,[.025,.975]).tolist(),
                per_seed=[float(m[:,0].sum()/m[:,1].sum()) for m in matrices],documents=len(a))


def main(fits,raw,out):
    out.mkdir(exist_ok=True,parents=True)
    records={}
    metrics={}
    curves=[]
    disc={}
    probes={}
    contexts={}
    efficiency={}
    for path in sorted(raw.glob('*.eval.json')):
        name=path.name.removesuffix('.eval.json')
        rec=read(path)
        train=read(fits/(name+'.train.json'))
        actual=hashlib.file_digest((fits/(name+'.pt')).open('rb'),'sha256').hexdigest()
        assert actual==rec['checkpoint_sha256']==train['checkpoint_sha256']
        assert all(np.isfinite(row).all() for row in train['train_curve'])
        records[name]=rec
        train['total_tokens']=train.get('total_tokens',train['tokens'])
        for step,loss,gradient in train.pop('train_curve'):
            curves.append([name,step,loss,gradient])
        res=metrics[name]=dict(training=train,validation_alpha=rec['alpha'],panels={})
        for key,rows in rec['rows'].items():
            n=sum(r['count'] for r in rows)
            score=sum(r['nll'] for r in rows)/n
            m=dict(targets=n,nll=score,ppl=float(np.exp(score)),correct=sum(r['top1'] for r in rows),
                   copy_hits=sum(r['hits'] for r in rows),copy_correct=sum(r['correct_copy'] for r in rows))
            if rec['alpha'] is not None:
                j=rec['grid'].index(rec['alpha']) if key.startswith('validation') else 0
                m['calibrated_cache_nll']=sum(r['mixture_nll'][j] for r in rows)/n
            res['panels'][key]=m
        f=raw/(name+'.discourse.json')
        if f.is_file():
            rows=read(f)
            assert len(rows)==len({r['passage_sha256'] for r in rows})==512
            val=dict(n=len(rows),exact=sum(r['exact'] for r in rows),total_nll=sum(r['nll'] for r in rows),
                     answer_subwords=sum(r['count'] for r in rows))
            if 'exact_copy_available' in rows[0]:
                val.update(exact_copy_available=sum(r['exact_copy_available'] for r in rows),
                           correct_with_full_copy=sum(r['exact'] and r['exact_copy_available'] for r in rows),
                           correct_with_prior_word=sum(r['exact'] and r['previous_word'] for r in rows))
                if rows[0]['mixture_exact'] is not None:
                    val['calibrated_cache_exact']=sum(r['mixture_exact'] for r in rows)
            disc[name]=val
        f=raw/(name+'.probe.json')
        if f.is_file():
            grouped=defaultdict(list)
            for r in read(f):
                grouped[r['kind']+'/'+str(r['distance'])].append(r)
            probes[name]={k:dict(n=len(rr),top1=sum(r['top']==r['target'] for r in rr),
                             preferred=sum(r['preferred'] for r in rr),
                             stale_top1=sum(r['top']==r['old'] for r in rr),
                             correct_index=sum(r['index_prediction']==r['target'] for r in rr))
                           for k,rr in grouped.items()}
        f=raw/(name+'.context.json')
        if f.is_file():
            rr=read(f)
            contexts[name]={k:sum(r[k] for r in rr)/sum(r['count'] for r in rr)
                            for k in rr[0] if '/' in k}
        f=raw/(name+'.bench.json')
        if f.is_file():
            rr=read(f)
            efficiency[name]={}
            for length in (128,512,1024):
                xs=[r for r in rr if r['length']==length]
                assert len(xs)==3 and all(r['generated_ids']==xs[0]['generated_ids'] for r in xs)
                efficiency[name][str(length)]=dict(prefill_ms=1000*statistics.median(r['prefill_s'] for r in xs),
                                  decode_ms_per_token=1000*statistics.median(r['decode_s'] for r in xs)/32,
                                  tensor_bytes=xs[0]['tensor_cache_bytes'],episode_bytes=xs[0]['episode_heap_estimate'])
    comparisons={}
    for tag,seeds in [('compact',(17,29,43)),('scale',(17,)),('extended',(17,))]:
        for right in (['local'] if tag=='extended' else ['local','transformer','mamba2']):
            for split in ('test','transfer'):
                for cache in (False,True):
                    k='/'.join((tag,'cog-minus-'+right+('-with-cache' if cache else ''),split))
                    comparisons[k]=paired(records,tag,seeds,'cog',right,split,cache)
    for right in ('slots','episode'):
        for split in ('test','transfer'):
            comparisons['compact/cog-minus-'+right+'/'+split]=paired(records,'compact',(17,29,43),'cog',right,split,False)
    with (out/'training_curves.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['fit','stage_step','training_nll','gradient_norm']);w.writerows(curves)
    for fname,data in [('metrics',metrics),('paired_intervals',comparisons),('discourse',disc),('retention',probes),
                       ('long_context',contexts),('efficiency',efficiency)]:
        write(out/(fname+'.json'),data)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('fits','raw','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.fits,a.raw,a.out)
