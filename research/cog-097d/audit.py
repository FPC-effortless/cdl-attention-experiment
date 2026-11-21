"""Independent accounting and data partition audit for 097d / 097d2.

No retraining; optional frozen neural checkpoint inference.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
from experiment import Encoder,make_corpus,TRAIN_MODULES,DEV_MODULES,TEST_MODULES,UNKNOWN_MODULES,rank_embedding,sha,scoring
from feedback import FRESH_MODULES
SEEDS=(1701,1707,1711)
ROOT=Path(__file__).parent

def audit(replay=False):
    docs={k:make_corpus(v) for k,v in [('train',TRAIN_MODULES),('dev',DEV_MODULES),
          ('test',TEST_MODULES),('unknown',UNKNOWN_MODULES),('fresh',FRESH_MODULES)]}
    def hashed(rows):return {sha(z['body']) for z in rows}
    # Exact train and fresh source overlaps are prohibited. 097d test contains
    # one original cross-module duplicate; experiment filtered it explicitly.
    assert not (hashed(docs['train']) & hashed(docs['fresh']))
    initial=set([z['query'].casefold() for z in docs['train']])
    assert not initial.intersection(z['query'].casefold() for z in docs['fresh'])
    counts={};results={}
    for phase,keyfile,key in [('d1','result-{}.json','results'),('d2','feedback-{}.json','scores')]:
        rows=[json.loads((ROOT/keyfile.format(seed)).read_text()) for seed in SEEDS]
        runs=[r[key] for r in rows]
        arms=tuple(runs[0].keys())
        assert all(tuple(run)==arms for run in runs)
        scores={a:{'correct_total':sum(x[a]['correct'] for x in runs),
                   'trial_total':sum(x[a]['n'] for x in runs),
                   'mean_accuracy':sum(x[a]['correct'] for x in runs)/sum(x[a]['n'] for x in runs),
                   'seed_counts':[x[a]['correct'] for x in runs]} for a in arms}
        for row,run in zip(rows,runs):
            assert all(s['n']==rows[0][key][a]['n'] for a,s in run.items())
            assert row['seed'] in SEEDS
            for arm in arms:
                s=run[arm]
                assert s['correct']==sum(x==1 for x in s['per_query_rank'])
                assert s['n']==len(s['per_query_rank'])
                assert abs(s['accuracy']-s['correct']/s['n'])<1e-5
        if phase=='d1':
            assert all(row['no_training_leak_tested'] for row in rows)
            assert all(row['counts']['test']==64 for row in rows)
            assert all(row['cross_split_duplicates_excluded']['test']>=1 for row in rows)
            assert all(row['training']['updates']==104 for row in rows)
        else:
            assert all(row['counts']['fresh_test']==161 for row in rows)
            assert all(row['feedback']['attempts']==591 for row in rows)
            assert all(row['feedback']['successes']==196 for row in rows)
            assert all(row['optimization']['updates']==152 for row in rows)
            assert all(row['scores']['after_feedback']['unknown_false_answers']==row['counts']['unknown'] for row in rows)
            for row in rows:
                assert len(row['feedback_trace'])==row['counts']['feedback_events']
                assert sum(x['reward'] for x in row['feedback_trace'])==196
        results[phase]=scores
        counts[phase]=rows[0]['counts']
    if replay:
        # Exact frozen model replay at authored test time, no optimizer updates.
        torch.set_num_threads(1)
        corpus=docs['fresh']
        tr=docs['train']
        blocked={sha(x['body']) for x in tr};qblocked={sha(x['query'].casefold()) for x in tr}
        corpus=[x for x in corpus if sha(x['body']) not in blocked and sha(x['query'].casefold()) not in qblocked]
        for seed in SEEDS:
            saved=torch.load(ROOT/f'feedback-{seed}.pt',map_location='cpu',weights_only=True)
            model=Encoder();model.load_state_dict(saved['state']);model.eval()
            s,_build,_queries=rank_embedding(corpus,model)
            expected=json.loads((ROOT/f'feedback-{seed}.json').read_text())['scores']['after_feedback']
            assert scoring(s)['per_query_rank']==expected['per_query_rank']
    return dict(status='audit passed',counts=counts,results=results,checkpoint_replayed=replay)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');p.add_argument('--out',default='audit-results.json')
    args=p.parse_args();out=audit(args.replay)
    (ROOT/args.out).write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))
