"""LRR-010: sealed pilot selection then 20 paired tuples/80 CPU fits.
No original scientific files are modified. TEST opens after ALL fits finish.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,resource,time,traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import multiprocessing as mp
import torch
from .lrr005_register import NAMES,FAMILIES
from .lrr008_invariance import InvariantRegister,SymbolicIndex,heldout,state_digest
from .lrr009_curriculum import make_bank,payload_hash,schedule,event_digest,predictions,summarize
from .lrr010_entropy import ARMS,CONDITIONS,MILESTONES,entropy_objective,dev_panels,dev_evaluate

PROTOCOL_COMMIT='04ed88d4f7635f576ceeec9dcb84ee9fd79eac3f'
ROOT=Path(__file__).resolve().parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()

def fit_one(model_seed,bank_seed,arm,weight,bank,hashes,panels,outdir,bank_size=500):
    torch.set_num_threads(1);torch.manual_seed(model_seed)
    model=InvariantRegister(); opt=torch.optim.AdamW(model.parameters(),lr=.004)
    init=state_digest(model); schedarm='interleaved' if arm.startswith('entropy') or arm=='interleaved_entropy' else arm
    events=schedule(schedarm,len(bank)); total=len(events)
    checkpoints={total//4,total//2,3*total//4,total}
    eval_steps=sorted({0,max(1,total//10),*checkpoints})
    train_time=0.;trace=[];dev=[];milestones=[];start=time.perf_counter()
    outdir.mkdir(parents=True,exist_ok=False)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},outdir/'initial.pt')
    with (outdir/'dev.jsonl').open('w') as log:
        rows,records=dev_evaluate(model.eval(),panels,0);dev+=rows
        for row in records:log.write(json.dumps(row)+'\n')
        for step,(idx,mode) in enumerate(events,1):
            examples,x,y,q,mask,noise=bank[idx]
            model.train();opt.zero_grad(set_to_none=True);t=time.perf_counter()
            loss,metrics=entropy_objective(model,x,y,q,mask,noise,mode,weight)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            train_time+=time.perf_counter()-t
            if step%50==0 or step==total:trace.append({'step':step,'bank_index':idx,'mode':mode,'loss':loss.item(),**metrics})
            if step in eval_steps:
                model.eval();rows,records=dev_evaluate(model,panels,step);dev+=rows
                for row in records:log.write(json.dumps(row)+'\n')
                log.flush()
            if step in checkpoints:
                counts=[int(v['step'].item()) for v in opt.state.values() if 'step' in v]
                assert counts and set(counts)=={step}
                receipt={'model_seed':model_seed,'bank_seed':bank_seed,'arm':arm,'entropy_weight':weight,
                         'step':step,'model_tensor_sha256':state_digest(model),'init_tensor_sha256':init,
                         'optimizer_step_min':min(counts),'optimizer_step_max':max(counts)}
                path=outdir/f'step{step}.pt'
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'receipt':receipt},path)
                milestones.append({**receipt,'filename':path.name,'sha256':sha(path)})
    assert [payload_hash(b) for b in bank]==hashes
    result={'model_seed':model_seed,'bank_seed':bank_seed,'arm':arm,'entropy_weight':weight,
            'parameters':sum(p.numel() for p in model.parameters()),'updates':total,
            'clean_updates':sum(m=='clean' for _,m in events),'augmented_updates':sum(m=='augmentation' for _,m in events),
            'question_view_exposures':total*48,'unique_source_histories':len(bank)*8,
            'init_tensor_sha256':init,'bank_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),
            'event_multiset_sha256':event_digest(events,hashes,sorted_events=True),
            'chronological_event_sha256':event_digest(events,hashes),
            'update_wall_seconds':train_time,'fit_plus_dev_wall_seconds':time.perf_counter()-start,
            'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'trace':trace,'milestones':milestones,'dev':dev,'start_time_utc':None,'finished_utc':utc()}
    write(outdir/'fit.json',result)
    return result


def tuple_worker(model_seed,bank_seed,arms,root,weight,bank_size=500,dev_trials=128,sem_per_cell=8):
    root=Path(root);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bank,hashes=make_bank(bank_seed,bank_size)
    panels=dev_panels(model_seed,dev_trials,sem_per_cell)
    write(root/f'bank_{model_seed}.json',{'model_seed':model_seed,'bank_seed':bank_seed,'hashes':hashes})
    receipts=[]
    for arm in arms:
        lam=float(arm.split('_')[1]) if arm.startswith('entropy_') else weight if arm=='interleaved_entropy' else 0.
        receipt=fit_one(model_seed,bank_seed,arm,lam,bank,hashes,panels,root/f'{arm}_{model_seed}',bank_size)
        receipts.append(receipt)
        print(json.dumps({'finished':arm,'seed':model_seed,'seconds':round(receipt['fit_plus_dev_wall_seconds'],2)}),flush=True)
    for key in ('init_tensor_sha256','bank_sha256','event_multiset_sha256'):
        assert len({r[key] for r in receipts})==1,(model_seed,key)
    return receipts


def freeze(root,phase,config):
    root.mkdir(parents=True,exist_ok=False)
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'relational_router').glob('*.py'))}
    rec={'experiment':'LRR-010','phase':phase,'protocol_commit':PROTOCOL_COMMIT,'config':config,
         'source_sha256':sources,'python':platform.python_version(),'torch':torch.__version__,
         'device':'local CPU; one PyTorch thread per worker','start_utc':utc()}
    write(root/'frozen_manifest.json',rec);return rec


def run_pool(root,pairs,arm_list,weight,workers=3,bank_size=500,dev_trials=128,sem_per_cell=8):
    receipts=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(tuple_worker,ms,bs,arm_list[i%len(arm_list):]+arm_list[:i%len(arm_list)],
                             str(root),weight,bank_size,dev_trials,sem_per_cell):(ms,bs)
                 for i,(ms,bs) in enumerate(pairs)}
        for fut in as_completed(futures):
            ms,bs=futures[fut]
            try:receipts.extend(fut.result())
            except BaseException:
                write(root/f'ERROR_{ms}.json',{'traceback':traceback.format_exc(),'pair':(ms,bs)});raise
            write(root/'fit_receipts.json',receipts)
    return receipts,time.perf_counter()-start


def final_dev(receipt,condition):
    return next(r for r in receipt['dev'] if r['step']==receipt['updates'] and r['condition']==condition)


def pilot(root,workers=3):
    cfg={'pairs':[(10901+i,20901+i) for i in range(3)],'lambdas':[0.,.02,.1,.3],
         'updates':1000,'selection':'nonzero: max qualified; min semantic collapse; max unfamiliar DEV; max regular DEV; min lambda'}
    seal=freeze(root,'pilot',cfg)
    rs,wall=run_pool(root,cfg['pairs'],['entropy_0','entropy_0.02','entropy_0.1','entropy_0.3'],0.,workers)
    options=[]
    for lam in cfg['lambdas']:
        rows=[r for r in rs if r['entropy_weight']==lam]
        options.append({'lambda':lam,'qualified':sum(final_dev(r,'regular')['recall']>=.9 for r in rows),
                        'collapsed':sum(final_dev(r,'semantic')['semantic_collapse'] for r in rows),
                        'mean_unfamiliar_dev':sum(final_dev(r,'untrained_in_vocab')['recall'] for r in rows)/3,
                        'mean_regular_dev':sum(final_dev(r,'regular')['recall'] for r in rows)/3})
    choice=max([o for o in options if o['lambda']>0],key=lambda o:(o['qualified'],-o['collapsed'],o['mean_unfamiliar_dev'],o['mean_regular_dev'],-o['lambda']))
    result={'selected_lambda':choice['lambda'],'options':options,'fits':len(rs),'wall_seconds':wall,
            'selected_utc':utc(),'test_evaluations':0}
    for name,digest in seal['source_sha256'].items():assert sha(ROOT/name)==digest
    write(root/'selection.json',result);print(json.dumps(result,indent=2),flush=True)


def negative(ex,kind):
    words=list(ex.words)
    if kind=='unknown_relation':words[-2]='lrr010_unknown_relation'
    else:
        start=ex.answer
        while start and words[start-1]!='.':start-=1
        assert words[start]==ex.subject
        words[start]=next(n for n in NAMES if n!=ex.subject)
        # Explicitly prove no matching entity/relation remains in the parsed history.
        clauses,entity,predicate=SymbolicIndex.parse(tuple(words))
        assert not any(name==entity and alias in FAMILIES[ex.family] for name,alias,pos in clauses)
    return replace(ex,words=tuple(words),answer=-1)


def evaluate_tuple(seed,bank_seed,root,trials=64,bank_size=500):
    root=Path(root);torch.set_num_threads(1);symbol=SymbolicIndex()
    # Exact same training bank, no hidden category-ID targets to symbolic learner.
    from .lrr008_invariance import augmented_batch
    for step in range(bank_size):
        batch=augmented_batch(bank_seed+2000,step,8)
        for ex in batch[0][:24]:symbol.observe(ex)
    assert len({symbol.root(w) for w in symbol.parents})==4
    cells=[];negcells=[]
    with (root/f'test_{seed}.jsonl').open('w') as poslog,(root/f'negative_{seed}.jsonl').open('w') as neglog:
        for arm in ARMS:
            model=InvariantRegister().eval();saved=torch.load(root/f'{arm}_{seed}'/'step1000.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(saved['model']);assert state_digest(model)==saved['receipt']['model_tensor_sha256']
            for n in (7,32,128):
                for cond in CONDITIONS:
                    case_seeds=[510_000_000_000+seed*1_000_000+n*1000+i for i in range(trials)]
                    examples=[heldout(cs,n,cond,split='test') for cs in case_seeds]
                    diag=predictions(model,examples)
                    refs=[symbol.route(ex) for ex in examples]
                    cell={'model_seed':seed,'bank_seed':bank_seed,'arm':arm,'facts':n,'condition':cond,
                          **summarize(diag),'symbolic_correct':sum(p==e.answer for p,e in zip(refs,examples))}
                    cells.append(cell)
                    for i,(ex,d,p) in enumerate(zip(examples,diag,refs)):
                        poslog.write(json.dumps({'model_seed':seed,'arm':arm,'facts':n,'condition':cond,'case':i,
                                                'case_seed':case_seeds[i],'gold':ex.answer,'symbolic_prediction':p,**d})+'\n')
                    if n==32 and cond=='regular':
                        for kind in ('known_absent','unknown_relation'):
                            negs=[negative(ex,kind) for ex in examples];dd=predictions(model,negs)
                            sym=[symbol.route(ex) for ex in negs];assert all(p is None for p in sym)
                            negcells.append({'model_seed':seed,'arm':arm,'kind':kind,'trials':trials,
                                             'false_reads':sum(d['pred'] is not None for d in dd)})
                            for i,(ex,d) in enumerate(zip(negs,dd)):
                                neglog.write(json.dumps({'model_seed':seed,'arm':arm,'kind':kind,'case':i,
                                                        'case_seed':case_seeds[i],'false_read':d['pred'] is not None,**d})+'\n')
            print(json.dumps({'evaluated':arm,'seed':seed}),flush=True)
    return {'model_seed':seed,'cells':cells,'negative_cells':negcells}


def confirm(root,selection_path,lock_commit,workers=3):
    choice=json.loads(Path(selection_path).read_text());weight=choice['selected_lambda']
    cfg={'pairs':[(11001+i,21001+i) for i in range(20)],'arms':ARMS,'entropy_lambda':weight,
         'lambda_lock_commit':lock_commit,'selection_file_sha256':sha(selection_path),
         'workers':workers,'updates':1000,'trials':64,'dev_trials':128,'semantic_trials':128}
    seal=freeze(root,'confirmation',cfg);start=time.perf_counter()
    fits,fitwall=run_pool(root,cfg['pairs'],list(ARMS),weight,workers)
    assert len(fits)==80
    write(root/'all_fits_closed.json',{'utc':utc(),'fits':80,'fit_wall_seconds':fitwall})
    print('ALL_80_FITS_CLOSED__TEST_OPENS',flush=True)
    cells=[];neg=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        tasks=[pool.submit(evaluate_tuple,ms,bs,str(root)) for ms,bs in cfg['pairs']]
        for f in as_completed(tasks):
            data=f.result();cells+=data['cells'];neg+=data['negative_cells']
    for p,digest in seal['source_sha256'].items():assert sha(ROOT/p)==digest,'source changed'
    result={'experiment':'LRR-010','config':cfg,'fits':fits,'cells':cells,'negative_cells':neg,
            'total_wall_seconds':time.perf_counter()-start,'fits_wall_seconds':fitwall,'finished_utc':utc()}
    write(root/'results.json',result);print('COMPLETED_LRR010',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('pilot','confirm'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--selection',type=Path);p.add_argument('--lock-commit')
    a=p.parse_args()
    if a.phase=='pilot':pilot(a.output,a.workers)
    else:
        if not a.selection or not a.lock_commit:raise ValueError('confirmation requires prospectively locked lambda')
        confirm(a.output,a.selection,a.lock_commit,a.workers)
