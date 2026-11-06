"""Frozen local CPU experiment: staged/alternating matched training order."""
from __future__ import annotations
import argparse, hashlib, json, platform, statistics, time
from collections import Counter
from pathlib import Path
from dataclasses import replace
import torch
from .lrr005_register import NAMES
from .lrr008_invariance import (InvariantRegister, SymbolicIndex, augmented_batch, triple,
    heldout, MAPS, FORBIDDEN, DEV_PAIR, TEST_PAIR, state_digest)
from .lrr009_curriculum import (ARMS, make_bank, fit_schedule, predictions, summarize, schedule)
PROTOCOL_COMMIT='f0274099f61aba0b5366514d3b4fdc5e7309a11b'
CONDITIONS=(*MAPS,'syntax_shift')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def absent(ex):
    words=list(ex.words);start=ex.answer
    while start and words[start-1]!='.':start-=1
    assert words[start]==ex.subject
    words[start]=next(name for name in NAMES if name!=ex.subject)
    return replace(ex,words=tuple(words),answer=-1)

def unknown(ex):
    words=list(ex.words);words[-2]='lrr009_unknown_relation'
    return replace(ex,words=tuple(words),answer=-1)

def retention(records,seed,half,total):
    selected=[r for r in records if r['arm']=='clean_first' and r['seed']==seed and r['condition']=='regular']
    before=sorted([r for r in selected if r['step']==half],key=lambda r:r['case'])
    after=sorted([r for r in selected if r['step']==total],key=lambda r:r['case'])
    assert [r['case'] for r in before]==[r['case'] for r in after]
    n=len(before);correct=sum(r['correct'] for r in before)
    lost=sum(a['correct'] and not b['correct'] for a,b in zip(before,after))
    gained=sum(not a['correct'] and b['correct'] for a,b in zip(before,after))
    return {'seed':seed,'warmup_trials':n,'warmup_correct':correct,
            'warmup_recall':correct/n,'qualified':correct/n>=.90,
            'final_dev_recall':sum(r['correct'] for r in after)/n,
            'lost_correct':lost,'new_gains':gained,
            'lost_fraction_of_initially_correct':lost/correct if correct else None,
            'retained_correct_fraction':1-lost/correct if correct else None}

def run(root,output,seeds=(71,72,73,74,75),bank_size=500,trials=64,dev_trials=128):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config={'seeds':list(seeds),'bank_size':bank_size,'updates':2*bank_size,
            'histories_per_minibatch':8,'question_views_per_minibatch':48,'trials':trials,
            'dev_trials':dev_trials,'arms':ARMS,'learning_rate':.004,'facts':[7,32,128],
            'conditions':CONDITIONS,'forbidden':sorted(FORBIDDEN),'test_pair':TEST_PAIR,
            'dev_pair':DEV_PAIR,'protocol_commit':PROTOCOL_COMMIT,
            'actual_execution':'local CPU working container; NOT GitHub Actions or external replication'}
    output.mkdir(parents=True,exist_ok=False)
    files={str(p.relative_to(root)):sha(p) for p in sorted((root/'relational_router').glob('*.py'))}
    manifest={'config':config,'files':files,'torch':torch.__version__,'python':platform.python_version(),
              'source_parent':'0c6d9ecd1a5a145a9d1b6a47b428313d29f44f8d','threads':1,'device':'CPU',
              'start_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    write(output/'frozen_manifest.json',manifest)
    report={'experiment':'LRR-009','config':config,'fits':[],'cells':[]}
    start=time.perf_counter();devrecords=[]
    checkpoints=output/'checkpoints'
    with (output/'dev_predictions.jsonl').open('w') as devlog:
        for seed in seeds:
            bank,hashes=make_bank(seed,bank_size)
            write(output/f'bank_receipt_{seed}.json',{'seed':seed,'packet_hashes':hashes,
                  'schedules':{a:schedule(a,bank_size) for a in ARMS}})
            for arm in ARMS:
                model,receipt,cases=fit_schedule(seed,arm,bank,hashes,checkpoints,dev_trials)
                report['fits'].append(receipt)
                for d in cases:
                    d={'seed':seed,'arm':arm,**d};devrecords.append(d)
                    devlog.write(json.dumps(d)+'\n')
                devlog.flush()
                write(output/'progress.json',report)
                print(json.dumps({'fit_finished':arm,'seed':seed,'wall':round(receipt['fit_plus_dev_wall_seconds'],2),
                    'final_dev_regular':next(r['recall'] for r in receipt['dev'] if r['step']==2*bank_size and r['condition']=='regular'),
                    'final_loss':receipt['trace'][-1]['loss']}),flush=True)
                del model
            paired=[r for r in report['fits'] if r['seed']==seed]
            for key in ('init_tensor_sha256','bank_sha256','event_multiset_sha256'):
                assert len({r[key] for r in paired})==1,(seed,key)
            assert len({r['chronological_event_sha256'] for r in paired})==3
            del bank
    report['fits_finished_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    write(output/'fit_receipts.json',report['fits'])
    # All fits are closed before the first trained-model TEST evaluation.
    print('ALL_FITS_COMPLETE__TEST_BEGINS',flush=True)
    with (output/'predictions.jsonl').open('w') as ledger:
        for seed in seeds:
            symbol=SymbolicIndex()
            for idx in range(bank_size):
                for i in range(8):
                    for ex in triple(8_000_000_000+(seed+2000)*1_000_000+idx*1000+i,idx,i):
                        symbol.observe(ex)
            assert len({symbol.root(w) for w in symbol.parents})==4
            for arm in ARMS:
                model=InvariantRegister().eval()
                ckpt=torch.load(checkpoints/f'lrr009_{arm}_{seed}_step{2*bank_size}.pt',weights_only=True,map_location='cpu')
                model.load_state_dict(ckpt['model'],strict=True)
                assert state_digest(model)==ckpt['receipt']['model_tensor_sha256']
                for n in (7,32,128):
                    normal=None
                    for cond in CONDITIONS:
                        seeds_case=[150_000_000_000+seed*1_000_000+n*1000+i for i in range(trials)]
                        exs=[heldout(cs,n,cond,split='test') for cs in seeds_case]
                        diag=predictions(model,exs)
                        for e,d in list(zip(exs,diag))[:4]:assert model.route(e.words)[0]==d['pred']
                        refs=[symbol.route(e) for e in exs]
                        if cond=='regular':normal=diag
                        row={'seed':seed,'arm':arm,'facts':n,'condition':cond,**summarize(diag),
                             'symbolic_correct':sum(p==e.answer for p,e in zip(refs,exs)),
                             'prediction_agreement':sum(a['pred']==b['pred'] for a,b in zip(normal,diag))/trials if cond!='syntax_shift' else None,
                             'both_correct':sum(a['correct'] and b['correct'] for a,b in zip(normal,diag))/trials}
                        if n==32 and cond=='regular':
                            row['known_absent_false_candidates']=sum(d['pred'] is not None for d in predictions(model,[absent(e) for e in exs]))
                            row['unknown_predicate_false_candidates']=sum(d['pred'] is not None for d in predictions(model,[unknown(e) for e in exs]))
                        for i,(e,d,p) in enumerate(zip(exs,diag,refs)):
                            ledger.write(json.dumps({'seed':seed,'arm':arm,'facts':n,'condition':cond,
                                  'case':i,'case_seed':seeds_case[i],'gold':e.answer,'symbolic_prediction':p,**d})+'\n')
                        report['cells'].append(row)
                print(json.dumps({'eval_finished':arm,'seed':seed,'primary':next(r['recall'] for r in report['cells'] if r['seed']==seed and r['arm']==arm and r['facts']==32 and r['condition']=='untrained_in_vocab')}),flush=True)
                write(output/'progress.json',report)
    def scores(arm,cond='untrained_in_vocab'):
        return [next(r['recall'] for r in report['cells'] if r['seed']==s and r['arm']==arm and r['facts']==32 and r['condition']==cond) for s in seeds]
    a=scores('interleaved');b=scores('clean_first');c=scores('augmentation_first');regular=scores('clean_first','regular')
    rt=[retention(devrecords,s,bank_size,2*bank_size) for s in seeds]
    qualified=[r for r in rt if r['qualified']];init=sum(r['warmup_correct'] for r in qualified);lost=sum(r['lost_correct'] for r in qualified)
    delta=[v-u for u,v in zip(a,b)]
    report['retention']=rt
    report['decision']={'interleaved_primary':a,'clean_first_primary':b,'augmentation_first_primary':c,
        'clean_first_minus_interleaved':delta,'primary_means':{arm:statistics.mean(scores(arm)) for arm in ARMS},
        'clean_first_regular':regular,'qualified_clean_first_seeds':[r['seed'] for r in qualified],
        'qualified_initial_correct':init,'qualified_lost_correct':lost,'qualified_lost_fraction':lost/init if init else None,
        'G1_capability':statistics.mean(b)>=.8 and sum(x>=.7 for x in b)>=4 and statistics.mean(regular)>=.9,
        'G2_order_uplift':statistics.mean(delta)>=.10 and min(delta)>=-.05,
        'G3_qualified_retention':len(qualified)>=4 and bool(init) and lost/init<=.05,
        'attention_replacement':'NOT TESTED','abstention':'diagnostic-only; not a core retrieval gate'}
    report['wall_seconds']=time.perf_counter()-start
    for p,h in files.items():assert sha(root/p)==h,'source changed during execution'
    write(output/'results.json',report)
    print(json.dumps(report['decision'],indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('evidence/run'))
    p.add_argument('--seeds',default='71,72,73,74,75');p.add_argument('--bank-size',type=int,default=500)
    p.add_argument('--trials',type=int,default=64);p.add_argument('--dev-trials',type=int,default=128)
    a=p.parse_args();run(Path(__file__).resolve().parent.parent,a.output,tuple(map(int,a.seeds.split(','))),a.bank_size,a.trials,a.dev_trials)
