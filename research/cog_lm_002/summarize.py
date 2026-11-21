"""Collapse paired raw receipts to per-fit scores, uncertainty and decision diagnostics."""
import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

ARMS=('mamba_lookup','cog_lookup','cog_revision','cog_revision_off')


def read(path):
    return json.loads(path.read_text())


def mean_loss(rows):
    return sum(r['nll'] for r in rows)/sum(r['count'] for r in rows)


def paired_loss(left,right):
    totals=collections.defaultdict(lambda:np.zeros(2))
    for x,y in zip(left,right,strict=True):
        assert (x['doc'],x['offset'],x['count'])==(y['doc'],y['offset'],y['count'])
        totals[x['doc']]+=np.array([x['nll']-y['nll'],x['count']])
    a=np.array(list(totals.values()))
    selections=np.random.default_rng(7142).integers(len(a),size=(5000,len(a)))
    sums=a[selections].sum(axis=1)
    diff=a.sum(axis=0)
    return dict(delta=diff[0]/diff[1],ci95=np.quantile(sums[:,0]/sums[:,1],[.025,.975]).tolist(),articles=len(a))


def probes(rows):
    groups=collections.defaultdict(list)
    for row in rows:
        groups[(row['kind'],row['delay'])].append(row)
    out={}
    for (kind,delay),g in groups.items():
        pairs=collections.defaultdict(list)
        for r in g:
            pairs[(r['template'],r['entity'])].append(r)
        assert all({r['flip'] for r in pair}=={0,1} for pair in pairs.values())
        stats=dict(n=len(g),top1=sum(r['top']==r['target'] for r in g),
            pairwise=sum(r['logodds']>0 for r in g),ties=sum(r['logodds']==0 for r in g),
            both_prefer=sum(all(r['logodds']>0 for r in pair) for pair in pairs.values()),
            both_top1=sum(all(r['top']==r['target'] for r in pair) for pair in pairs.values()),
            n_pairs=len(pairs),exact_lookup=sum(r['lookup']==r['target'] for r in g),
            mean_logodds=float(np.mean([r['logodds'] for r in g])))
        if any('value_max_delta' in r for r in g):
            stats['largest_bank_value_change']=max(r.get('value_max_delta',0.) for r in g)
        out[f'{kind}/{delay}']=stats
    return out


def dialogues(rows):
    before = np.array([r['before_new_logp']-r['before_old_logp'] for r in rows])
    after = np.array([r['after_new_logp']-r['after_old_logp'] for r in rows])
    return dict(cases=len(rows),dialogues=len({r['dialogue'] for r in rows}),
        before_prefers_old=int(sum(before<0)),after_prefers_new=int(sum(after>0)),
        switched_correctly=int(sum((before<0)&(after>0))),switched_backwards=int(sum((before>0)&(after<0))),
        positive_shift=int(sum(after>before)),median_logodds_shift=float(np.median(after-before)),
        exact_after=sum(r['after_new_exact'] for r in rows),
        literal_after=sum(r['after_literal_prefers_new'] for r in rows),
        literal_switch=sum(not r['before_literal_prefers_new'] and r['after_literal_prefers_new'] for r in rows))


def build(root,joint):
    result=dict(scores={},paired={},probes={},contexts={},costs={},fits={},dialogues={},matched={})
    for prefix,train_root in [('',root),('joint_',joint)]:
        for arm in ARMS:
            name=prefix+arm
            scores={}
            for split in ('test','transfer'):
                rows=read(root/(name+'.'+split+'.json'))
                scores[split]=dict(nll=mean_loss(rows),targets=sum(r['count'] for r in rows))
            new=read(root/(name+'.lambada.json'))
            scores['lambada']=dict(n=len(new),exact=sum(r['exact'] for r in new),
                exact_with_copy=sum(r['exact'] and r['available'] for r in new),final_word_token_nll=mean_loss(new))
            result['scores'][name]=scores
            result['probes'][name]=probes(read(root/(name+'.revisions.json')))
            context=read(root/(name+'.context.json'))
            result['contexts'][name]={str(d):mean_loss([r for r in context if r['distance']==d]) for d in (256,1024,4096,16384)}
            if not arm.endswith('_off'):
                f=read(train_root/(name+'.json'))
                result['fits'][name]={k:v for k,v in f.items() if k!='loss'}
            if prefix == 'joint_':
                result['dialogues'][name]=dialogues(read(root/(name+'.dialogue.json')))
            if arm == 'cog_revision':
                probes_matched=read(root/(name+'.matched.json'))
                result['matched'][name]=dict(windows=len(probes_matched),
                    masked_disabled_max_delta=max(r['masked_disabled_max_delta'] for r in probes_matched),
                    full_nll=sum(r['full_nll'] for r in probes_matched)/(128*len(probes_matched)),
                    masked_nll=sum(r['masked_nll'] for r in probes_matched)/(128*len(probes_matched)))
            cost=root/(name+'.cost.json')
            if cost.exists():
                record=read(cost)
                summary={k:v for k,v in record.items() if k!='measurements'}
                for d in (256,1024,4096,16384):
                    ms=[r for r in record['measurements'] if r['request_length']==d]
                    summary[str(d)]={k:statistics.median(r[k] for r in ms) for k in
                        ('encoded','encoding_s','prefill_s','decode_s','request_s','cache_tensor_bytes','table_heap_bytes','table_entries')}
                result['costs'][name]=summary
        for arm in ('mamba_lookup','cog_lookup','cog_revision_off'):
            for split in ('test','transfer'):
                rows=read(root/(prefix+'cog_revision.'+split+'.json'))
                controls=read(root/(prefix+arm+'.'+split+'.json'))
                result['paired'][prefix+arm+'/'+split]=paired_loss(rows,controls)
    gap=read(root/'scale-mamba2-17-continued.eval.json')
    result['gap']={'alpha':gap['alpha'],'checkpoint_sha256':gap['checkpoint_sha256']}
    for split in ('test','transfer'):
        rows=gap['rows'][split+'/full']
        result['gap'][split]=dict(plain=mean_loss(rows),lookup=sum(r['mixture_nll'][0] for r in rows)/sum(r['count'] for r in rows))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('raw','joint','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(json.dumps(build(a.raw,a.joint),indent=2)+'\n')
