"""Prospectively sealed LRR-011, 20 paired tuples / 60 actual local CPU fits.

A separate TEST phase opens only after every fit closes. No model selection,
no dropped seeds, no reset/annealing, and no changed hard runtime.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,resource,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import multiprocessing as mp
import torch
from .lrr005_register import vectorize,ID
from .lrr008_invariance import InvariantRegister,SymbolicIndex,heldout,state_digest,augmented_batch
from .lrr009_curriculum import make_bank,payload_hash,schedule,event_digest,predictions,summarize
from .lrr010_entropy import CONDITIONS,dev_panels,dev_evaluate
from .run_lrr010 import negative
from .lrr011_binding import ARMS,WEIGHT,binding_objective

PROTOCOL_COMMIT='9baa328256709a6d22542511cf09f74f52c64905'
ROOT=Path(__file__).resolve().parent.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()

@torch.no_grad()
def read_write_diagnostics(model,examples):
    """Additional evaluator-only soft/hard contrast; not a service scorer."""
    output=[]
    for start in range(0,len(examples),32):
        exs=examples[start:start+32];x,y,q=vectorize(exs)
        e,r,g=model(x);rows=torch.arange(len(x))
        qr,gr=r[rows,q],r[rows,y]
        # TRAIN-like readout as a diagnostic: includes a null logit but no
        # last-write policy; it is NOT equivalent to the hard index algorithm.
        score=((e*e[rows,q,None]).sum(-1)*(r*qr[:,None]).sum(-1)*g[:,:,2]).clamp_min(1e-9)
        valid=torch.arange(x.shape[1])[None]<q[:,None]
        logits=torch.where(valid,score.log(),torch.full_like(score,-1e9))
        logits=torch.cat((logits,model.null_logit.expand(len(x),1)),-1)
        softpred=logits.argmax(-1).tolist()
        h=model.token_features(x);gg=model.role(h).sigmoid().tolist();cc=model.code(h).argmax(-1).tolist()
        for j,ex in enumerate(exs):
            entity=relation=None;gold_entity=gold_relation=None
            for pos,word in enumerate(ex.words[:-1]):
                if pos==ex.answer:gold_entity,gold_relation=entity,relation
                if gg[j][pos][0]>.5:entity=word
                if gg[j][pos][1]>.5:relation=cc[j][pos]
            ent=lambda p:float(-(p*p.clamp_min(1e-9).log()).sum())
            output.append({'soft_read_write_agreement':float((qr[j]*gr[j]).sum()),
                           'query_maxprob':float(qr[j].max()),'gold_maxprob':float(gr[j].max()),
                           'query_soft_entropy':ent(qr[j]),'gold_soft_entropy':ent(gr[j]),
                           'hard_relation_equal':relation is not None and relation==gold_relation,
                           'hard_entity_equal':entity is not None and entity==gold_entity,
                           'soft_pointer':None if softpred[j]==x.shape[1] else softpred[j],
                           'soft_pointer_correct':softpred[j]==ex.answer})
    return output


def fit_one(model_seed,bank_seed,arm,bank,hashes,panels,outdir):
    torch.set_num_threads(1);torch.manual_seed(model_seed)
    model=InvariantRegister();opt=torch.optim.AdamW(model.parameters(),lr=.004)
    init=state_digest(model);events=schedule('interleaved',len(bank));total=len(events)
    cksteps={total//4,total//2,3*total//4,total}
    evalsteps=sorted({0,max(1,total//10),*cksteps})
    outdir.mkdir(parents=True,exist_ok=False)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},outdir/'initial.pt')
    trace=[];dev=[];checkpoints=[];updatewall=0.;started=time.perf_counter();startutc=utc()
    with (outdir/'dev.jsonl').open('w') as out:
        ss,rr=dev_evaluate(model.eval(),panels,0);dev+=ss
        for row in rr:out.write(json.dumps(row)+'\n')
        for step,(idx,mode) in enumerate(events,1):
            examples,x,y,q,mask,noise=bank[idx]
            model.train();opt.zero_grad(set_to_none=True);t=time.perf_counter()
            loss,metrics=binding_objective(model,x,y,q,mask,noise,mode,arm,WEIGHT)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss {arm} {model_seed} {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            updatewall+=time.perf_counter()-t
            if step%50==0 or step==total:trace.append({'step':step,'bank_index':idx,'mode':mode,'loss':loss.item(),**metrics})
            if step in evalsteps:
                model.eval();ss,rr=dev_evaluate(model,panels,step);dev+=ss
                for row in rr:out.write(json.dumps(row)+'\n')
                out.flush()
            if step in cksteps:
                counters=[int(s['step'].item()) for s in opt.state.values() if 'step' in s]
                assert counters and set(counters)=={step}
                receipt={'model_seed':model_seed,'bank_seed':bank_seed,'arm':arm,'step':step,
                         'model_tensor_sha256':state_digest(model),'init_tensor_sha256':init,
                         'optimizer_step_min':min(counters),'optimizer_step_max':max(counters)}
                path=outdir/f'step{step}.pt'
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'receipt':receipt},path)
                checkpoints.append({**receipt,'filename':path.name,'sha256':sha(path)})
    assert [payload_hash(b) for b in bank]==hashes
    result={'model_seed':model_seed,'bank_seed':bank_seed,'arm':arm,'aux_weight':WEIGHT if arm!='control' else 0.,
            'parameters':sum(p.numel() for p in model.parameters()),'updates':total,
            'clean_updates':len(bank),'augmented_updates':len(bank),'question_view_exposures':total*48,
            'unique_source_histories':len(bank)*8,'init_tensor_sha256':init,
            'bank_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),
            'event_multiset_sha256':event_digest(events,hashes,sorted_events=True),
            'chronological_event_sha256':event_digest(events,hashes),
            'update_wall_seconds':updatewall,'fit_plus_dev_wall_seconds':time.perf_counter()-started,
            'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'started_utc':startutc,'finished_utc':utc(),'trace':trace,'milestones':checkpoints,'dev':dev}
    write(outdir/'fit.json',result)
    return result


def tuple_worker(ms,bs,order,root,bank_size=500,dev_trials=128,per_cell=8):
    root=Path(root);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bank,hashes=make_bank(bs,bank_size);panels=dev_panels(ms,dev_trials,per_cell)
    write(root/f'bank_{ms}.json',{'model_seed':ms,'bank_seed':bs,'hashes':hashes})
    outputs=[]
    for arm in order:
        r=fit_one(ms,bs,arm,bank,hashes,panels,root/f'{arm}_{ms}')
        outputs.append(r)
        print(json.dumps({'fit_finished':arm,'seed':ms,'seconds':round(r['fit_plus_dev_wall_seconds'],2)}),flush=True)
    for key in ('init_tensor_sha256','bank_sha256','event_multiset_sha256','chronological_event_sha256'):
        assert len({r[key] for r in outputs})==1,(key,ms)
    return outputs


def eval_tuple(ms,bs,root,trials=64,bank_size=500):
    root=Path(root);torch.set_num_threads(1)
    symbol=SymbolicIndex()
    for step in range(bank_size):
        batch=augmented_batch(bs+2000,step,8)
        for ex in batch[0][:24]:symbol.observe(ex)
    assert len({symbol.root(w) for w in symbol.parents})==4
    cells=[];negcells=[];diagnostics=[]
    with (root/f'test_{ms}.jsonl').open('w') as po,(root/f'negative_{ms}.jsonl').open('w') as no, (root/f'binding_{ms}.jsonl').open('w') as bio:
        for arm in ARMS:
            model=InvariantRegister().eval()
            saved=torch.load(root/f'{arm}_{ms}'/'step1000.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(saved['model']);assert state_digest(model)==saved['receipt']['model_tensor_sha256']
            for n in (7,32,128):
                seeds=[510_000_000_000+ms*1_000_000+n*1000+i for i in range(trials)]
                for cond in CONDITIONS:
                    exs=[heldout(s,n,cond,split='test') for s in seeds]
                    pred=predictions(model,exs);refs=[symbol.route(e) for e in exs]
                    assert all(a==e.answer for a,e in zip(refs,exs))
                    cells.append({'model_seed':ms,'bank_seed':bs,'arm':arm,'facts':n,'condition':cond,
                                  **summarize(pred),'symbolic_correct':sum(a==e.answer for a,e in zip(refs,exs))})
                    for i,(ex,pp,ref) in enumerate(zip(exs,pred,refs)):
                        po.write(json.dumps({'model_seed':ms,'arm':arm,'facts':n,'condition':cond,'case':i,
                                             'case_seed':seeds[i],'gold':ex.answer,'symbolic_prediction':ref,**pp})+'\n')
                    if n==32 and cond=='regular':
                        for kind in ('known_absent','unknown_relation'):
                            nn=[negative(e,kind) for e in exs];dd=predictions(model,nn)
                            assert all(symbol.route(e) is None for e in nn)
                            negcells.append({'model_seed':ms,'arm':arm,'kind':kind,'trials':trials,
                                             'false_reads':sum(d['pred'] is not None for d in dd)})
                            for i,d in enumerate(dd):
                                no.write(json.dumps({'model_seed':ms,'arm':arm,'kind':kind,'case':i,'case_seed':seeds[i],
                                                     'false_read':d['pred'] is not None,**d})+'\n')
                    if n==32 and cond in ('regular','untrained_in_vocab'):
                        ds=read_write_diagnostics(model,exs)
                        diagnostics.append({'model_seed':ms,'arm':arm,'condition':cond,'trials':trials,
                                            **{k:sum(d[k] for d in ds)/trials for k in ds[0] if k!='soft_pointer'}})
                        for i,d in enumerate(ds):
                            bio.write(json.dumps({'model_seed':ms,'arm':arm,'condition':cond,'case':i,'case_seed':seeds[i],
                                                  'gold':exs[i].answer,'hard_pred':pred[i]['pred'],**d})+'\n')
            print(json.dumps({'evaluated':arm,'seed':ms}),flush=True)
    return {'cells':cells,'negative_cells':negcells,'binding_diagnostics':diagnostics}


def run(root,workers=3):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    pairs=[(12001+i,22001+i) for i in range(20)]
    config={'pairs':pairs,'arms':ARMS,'weight':WEIGHT,'updates':1000,'bank_size':500,'trials':64,
            'dev_trials':128,'semantic_trials':128,'workers':workers,'model_parameters':4748}
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'relational_router').glob('*.py'))}
    seal={'experiment':'LRR-011','protocol_commit':PROTOCOL_COMMIT,'start_utc':utc(),'config':config,
          'source_sha256':sources,'python':platform.python_version(),'torch':torch.__version__,
          'device':'local CPU; one PyTorch thread per worker'}
    write(root/'frozen_manifest.json',seal);start=time.perf_counter();fits=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(tuple_worker,ms,bs,list(ARMS[i%3:]+ARMS[:i%3]),str(root)):(ms,bs)
                 for i,(ms,bs) in enumerate(pairs)}
        for f in as_completed(futures):
            try:fits.extend(f.result())
            except BaseException:
                write(root/f'ERROR_{futures[f][0]}.json',{'traceback':traceback.format_exc(),'tuple':futures[f]});raise
            write(root/'fit_receipts.json',fits)
    assert len(fits)==60
    fitwall=time.perf_counter()-start
    write(root/'all_fits_closed.json',{'fits':60,'utc':utc(),'fit_wall_seconds':fitwall})
    print('ALL_60_FITS_CLOSED__TEST_OPENS',flush=True)
    cells=[];neg=[];diag=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        tasks=[pool.submit(eval_tuple,ms,bs,str(root)) for ms,bs in pairs]
        for f in as_completed(tasks):
            d=f.result();cells+=d['cells'];neg+=d['negative_cells'];diag+=d['binding_diagnostics']
    for path,digest in sources.items():assert sha(ROOT/path)==digest,('source changed',path)
    output={'experiment':'LRR-011','config':config,'fits':fits,'cells':cells,'negative_cells':neg,
            'binding_diagnostics':diag,'fit_wall_seconds':fitwall,'total_wall_seconds':time.perf_counter()-start,
            'finished_utc':utc()}
    write(root/'results.json',output);print('COMPLETED_LRR011',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();run(a.output,a.workers)
