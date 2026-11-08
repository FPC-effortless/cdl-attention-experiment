"""Frozen VB-002 runner. No validation tuning; all fits precede held-out reads."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from ..versioned_baselines.study import Config, dataset, digest_data, take
from .model import SentenceLRC, support_loss, gold_route_overrides

ARMS = ('lrc_answer_cf', 'lrc_support_cf', 'lrc_oracle_control')
FIELDS = ('write_relation','write_address','routes','query_first','rank_first',
          'page_first','query_second','rank_second','page_second','continuation')


def hashes():
    root=Path(__file__).resolve().parents[1]
    paths=['hierarchical_model.py','versioned_lrc/model.py','versioned_lrc/run.py',
           'versioned_lrc/diagnostics.py','versioned_baselines/study.py']
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def per_loss(out, answer):
    return F.cross_entropy(out.logits[:,-1],answer,reduction='none')


def intervention(base, rng):
    key = (rng.randrange(2),rng.choice(FIELDS))
    actions=base.decisions[key]
    changed=actions.clone()
    policy=base.policies[key]
    coords=[]
    for row in range(actions.shape[0]):
        t=rng.randrange(actions.shape[1]-1) if key[1].startswith('write_') else actions.shape[1]-1
        coord=(row,t) if actions.ndim==2 else (row,t,rng.randrange(actions.shape[2]))
        old=int(actions[coord])
        alt=rng.randrange(policy.shape[-1]-1)
        alt+=int(alt>=old)
        changed[coord]=alt
        coords.append((coord,old,alt))
    return key,changed,coords


def fit(out, config, train, arm, seed):
    torch.manual_seed(seed)
    model=SentenceLRC()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=0.0)
    order=torch.Generator().manual_seed(100000+seed)
    rng=random.Random(900000+seed)
    trace=[]
    counters={field:dict(trials=0,nonzero=0,operand_changed=0,sum_abs_delta=0.0) for field in FIELDS}
    batch_hash=hashlib.sha256()
    start=time.perf_counter()
    for step in range(1,config.steps+1):
        idx=torch.randint(len(train['answer']),(config.batch,),generator=order)
        batch_hash.update(idx.numpy().tobytes())
        b=take(train,idx)
        optimizer.zero_grad(set_to_none=True)
        auxiliary=torch.tensor(0.)
        credit=torch.tensor(0.)
        if arm=='lrc_oracle_control':
            with torch.no_grad():
                ordinary=model(b['records'],b['query'])
                forced=gold_route_overrides(ordinary,b['support'])
            base=model(b['records'],b['query'],overrides=forced)
            assert all(torch.equal(v[:,-1],b['support']) for v in base.operands)
            nll=per_loss(base,b['answer']).mean()
            objective=nll
        else:
            base=model(b['records'],b['query'])
            loss=per_loss(base,b['answer'])
            nll=loss.mean()
            key,changed,coords=intervention(base,rng)
            with torch.no_grad():
                alt=model(b['records'],b['query'],overrides={key:changed})
                delta=loss.detach()-per_loss(alt,b['answer'])
                original_refs=torch.stack([v[:,-1] for v in base.operands],1)
                alt_refs=torch.stack([v[:,-1] for v in alt.operands],1)
                c=counters[key[1]]
                c['trials']+=len(delta)
                c['nonzero']+=int(delta.abs().gt(1e-7).sum())
                c['sum_abs_delta']+=float(delta.abs().sum())
                c['operand_changed']+=int(original_refs.ne(alt_refs).any(-1).any(-1).sum())
            logp=base.policies[key].log_softmax(-1)
            differences=torch.stack([logp[coord][old]-logp[coord][alternative] for coord,old,alternative in coords])
            credit=(delta.clamp(-1,1)*differences).mean()
            if arm=='lrc_support_cf':
                auxiliary=support_loss(base,b['support'])
            objective=nll+credit+auxiliary
        if not torch.isfinite(objective):
            raise RuntimeError('nonfinite objective')
        objective.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad):
            raise RuntimeError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%100==0:
            trace.append(dict(step=step,nll=float(nll.detach()),credit=float(credit.detach()),support=float(auxiliary.detach())))
        if step%500==0:
            print(json.dumps(dict(event='train_progress',arm=arm,seed=seed,**trace[-1])),flush=True)
    receipt=dict(arm=arm,seed=seed,steps=config.steps,forward_passes=2*config.steps,
                 train_seconds=time.perf_counter()-start,trace=trace,
                 batch_order_sha256=batch_hash.hexdigest(),credit_counts=counters,source_hashes=hashes())
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                    order_rng=order.get_state(),intervention_rng=rng.getstate(),receipt=receipt),out/f'{arm}_{seed}.pt')
    (out/f'{arm}_{seed}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(event='fit_complete',arm=arm,seed=seed,seconds=receipt['train_seconds'],last=trace[-1])),flush=True)


@torch.inference_mode()
def evaluate(model,data,oracle=False,memory='normal'):
    predictions=[];losses=[];operands=[];admissions=[]
    for start in range(0,len(data['answer']),64):
        b=take(data,slice(start,start+64))
        base=model(b['records'],b['query'],memory=memory)
        if oracle:
            base=model(b['records'],b['query'],overrides=gold_route_overrides(base,b['support']))
        predictions.append(base.logits[:,-1].argmax(-1))
        losses.append(per_loss(base,b['answer']))
        operands.append(torch.stack([v[:,-1] for v in base.operands],1))
        gold=gold_route_overrides(base,b['support'])
        per_layer=[]
        for ell in range(2):
            hits=[]
            for hop,name in enumerate(('first','second')):
                route=base.decisions[(ell,'routes')][:,-1,hop].eq(gold[(ell,'routes')][:,-1,hop])
                addr=base.decisions[(ell,f'query_{name}')][:,-1].eq(gold[(ell,f'query_{name}')][:,-1])
                active=base.active_page_bits[(ell,f'page_{name}')][:,-1]
                page=base.decisions[(ell,f'page_{name}')][:,-1]
                target=gold[(ell,f'page_{name}')][:,-1]
                same_page=((page*active)==target).all(-1)
                hits.append(route & addr & same_page)
            per_layer.append(torch.stack(hits,-1))
        admissions.append(torch.stack(per_layer,1))
    pred,loss,selected,hit=map(torch.cat,(predictions,losses,operands,admissions))
    correct=pred.eq(data['answer'])
    matched=selected.eq(data['support'][:,None,:])
    result=dict(accuracy=float(correct.float().mean()),nll=float(loss.mean()),
                quartet_accuracy=float(correct.reshape(-1,4).all(-1).float().mean()),
                old_accuracy=float(correct[data['era'].eq(0)].float().mean()),
                latest_accuracy=float(correct[data['era'].eq(1)].float().mean()),
                operand_accuracy=matched.float().mean(0).tolist(),candidate_hit=hit.float().mean(0).tolist(),
                final_both=float(matched[:,-1].all(-1).float().mean()))
    condition=hit[:,-1].all(-1)
    result['both_final_candidates_count']=int(condition.sum())
    result['accuracy_given_both_final_candidates']=float(correct[condition].float().mean()) if condition.any() else None
    return result,dict(pred=pred,loss=loss,selected=selected,admitted=hit)


def test_all(out,config):
    for arm in ARMS:
        for seed in config.seeds:
            assert (out/f'{arm}_{seed}.pt').exists()
    tests=dict(iid=dataset(config.iid_seed,config.test_groups),heldout_pairs=dataset(config.heldout_seed,config.test_groups,True))
    results=[];predictions={}
    for seed in config.seeds:
        for arm in ARMS:
            model=SentenceLRC().eval()
            model.load_state_dict(torch.load(out/f'{arm}_{seed}.pt',weights_only=True)['model'])
            item=dict(arm=arm,seed=seed)
            for name,data in tests.items():
                for suffix,kwargs in (('',dict(oracle=arm=='lrc_oracle_control')),
                                      ('_learned',{}),('_memory_off',dict(memory='disabled')),
                                      ('_oracle_diagnostic',dict(oracle=True))):
                    key=name+suffix
                    item[key],predictions[f'{arm}/{seed}/{key}']=evaluate(model,data,**kwargs)
            results.append(item)
    torch.save(predictions,out/'predictions.pt')
    torch.save(tests,out/'evaluation_data.pt')
    summary=dict(results=results,evaluation_sha256={k:digest_data(v) for k,v in tests.items()},
                 completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='evaluation_complete',results=results)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg=Config()
    args.out.mkdir(parents=True,exist_ok=False)
    train=dataset(cfg.train_seed,cfg.train_groups)
    torch.save(train,args.out/'train.pt')
    manifest=dict(config=asdict(cfg),arms=ARMS,source_hashes=hashes(),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  train_sha256=digest_data(train),parameters=sum(p.numel() for p in SentenceLRC().parameters()),
                  torch=torch.__version__,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
    for seed in cfg.seeds:
        for arm in ARMS:
            fit(args.out,cfg,train,arm,seed)
    test_all(args.out,cfg)

if __name__=='__main__':
    main()
