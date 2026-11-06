"""Frozen LRR-008 local CPU run; saves every fit, optimizer and case prediction."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import statistics
import torch
from .lrr005_register import FAMILIES, NAMES
from .lrr008_invariance import *

BASE_BLOB="fb397a428ff483ce89ec14128fee9437487115a5"
PROTOCOL_COMMIT="c7450e12dbd36b36049412c3b1c208e4ba67d688"
CONDITIONS=(*MAPS,"syntax_shift")

def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def freeze(root,output,config):
    src=root/"relational_router"
    b=(src/"lrr005_register.py").read_bytes()
    actual=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if actual!=BASE_BLOB:raise RuntimeError("unreviewed original runtime")
    manifest={"protocol_commit":PROTOCOL_COMMIT,"source_parent":"09becc095010e14002576a3582e8577aa54eb690",
              "configuration":config,"files":{str(p.relative_to(root)):fingerprint(p) for p in sorted(src.glob('*.py'))},
              "torch":torch.__version__,"python":platform.python_version(),"device":"CPU","threads":1,
              "wall_clock_start_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              "execution":"active local working container; NOT GitHub Actions or independent external replication"}
    output.mkdir(parents=True,exist_ok=True);json_write(output/"frozen_manifest.json",manifest)
    return manifest

def absent(ex):
    words=list(ex.words);start=ex.answer
    while start and words[start-1]!='.':start-=1
    assert words[start]==ex.subject
    words[start]=next(n for n in NAMES if n!=ex.subject)
    return replace(ex,words=tuple(words),answer=-1)

def unknown(ex):
    words=list(ex.words);words[-2]='lrr8unseenpredicate'
    return replace(ex,words=tuple(words),answer=-1)

def run(root,output,seeds=(61,62,63,64,65),steps=500,trials=64):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config={"seeds":list(seeds),"steps":steps,"trials":trials,"arms":ARMS,"facts":[7,32,128],
            "conditions":CONDITIONS,"histories_per_update":8,"learning_rate":.004,"invariance_weight":.5,
            "augmentation":"second-view generator-labeled filler embeddings replaced by N(0,I)*U[.5,1.5]",
            "test_pair":TEST_PAIR,"dev_pair":DEV_PAIR,"forbidden_pairs":sorted(FORBIDDEN)}
    manifest=freeze(root,output,config);report={"experiment":"LRR-008","config":config,"training":[],"cells":[]}
    ckptdir=output/"checkpoints";total_start=time.perf_counter()
    # All models finish before any trained-model TEST is inspected.
    for seed in seeds:
        for arm in ARMS:
            model,receipt=fit(seed,arm,steps=steps,histories=8,outdir=ckptdir)
            report['training'].append(receipt)
            print(json.dumps({"fit_finished":arm,"seed":seed,"seconds":round(receipt['wall_seconds'],2),
                              "last_loss":receipt['trace'][-1]['loss']}),flush=True)
            json_write(output/'progress.json',report)
            del model
        paired=[r for r in report['training'] if r['seed']==seed]
        for field in ('init_tensor_sha256','input_stream_sha256','augmentation_sha256'):
            assert len({r[field] for r in paired})==1, f"unmatched {field} seed {seed}"
    json_write(output/'fit_receipts.json',report['training'])
    with (output/'predictions.jsonl').open('w') as ledger:
        for seed in seeds:
            symbolic=SymbolicIndex()
            for step in range(steps):
                for i in range(8):
                    for ex in triple(8_000_000_000+seed*1_000_000+step*1000+i,step,i):symbolic.observe(ex)
            assert len({symbolic.root(k) for k in symbolic.parents})==4
            for arm in ARMS:
                model=InvariantRegister().eval()
                ckpt=torch.load(ckptdir/f'lrr008_{arm}_{seed}.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(ckpt['model'],strict=True)
                assert state_digest(model)==ckpt['receipt']['final_tensor_sha256']
                for n in (7,32,128):
                    reg_predictions=None;reg_correct=None
                    for condition in CONDITIONS:
                        case_seeds=[90_000_000_000+seed*1_000_000+n*1000+i for i in range(trials)]
                        examples=[heldout(s,n,condition) for s in case_seeds]
                        diag=hard_batch(model,examples)
                        # Independent original inference helper checked on first four cases of EACH cell.
                        for ex,d in list(zip(examples,diag))[:4]:assert model.route(ex.words)[0]==d['pred']
                        refs=[symbolic.route(ex) for ex in examples]
                        failures=Counter(d['failure'] for d in diag)
                        hits=sum(d['correct'] for d in diag)
                        if condition=='regular':
                            reg_predictions=[d['pred'] for d in diag];reg_correct=[d['correct'] for d in diag]
                        samepos=condition!='syntax_shift'
                        row={"arm":arm,"seed":seed,"facts":n,"condition":condition,"trials":trials,
                             "correct":hits,"recall":hits/trials,"failures":dict(failures),
                             "symbolic_correct":sum(p==ex.answer for p,ex in zip(refs,examples)),
                             "paired_both_correct":sum(a and d['correct'] for a,d in zip(reg_correct,diag))/trials,
                             "prediction_agreement":sum(p==d['pred'] for p,d in zip(reg_predictions,diag))/trials if samepos else None,
                             "mean_writes":statistics.mean(d['writes'] for d in diag),
                             "mean_nonvalue_writes":statistics.mean(d['nonvalue_writes'] for d in diag),
                             "mean_collision_pairs":statistics.mean(d['collision_pairs'] for d in diag),
                             "gold_written_fraction":statistics.mean(d['gold_written'] for d in diag)}
                        if n==32 and condition=='regular':
                            row['known_absent_false_candidates']=sum(d['pred'] is not None for d in hard_batch(model,[absent(ex) for ex in examples]))
                            row['unknown_predicate_false_candidates']=sum(d['pred'] is not None for d in hard_batch(model,[unknown(ex) for ex in examples]))
                        for i,(ex,d) in enumerate(zip(examples,diag)):
                            ledger.write(json.dumps({"arm":arm,"seed":seed,"facts":n,"condition":condition,
                                                     "case_seed":case_seeds[i],"case":i,"gold":ex.answer,
                                                     "symbolic_prediction":refs[i],**d})+'\n')
                        report['cells'].append(row)
                print(json.dumps({"eval_finished":arm,"seed":seed,"primary_recall":next(r['recall'] for r in report['cells'] if r['seed']==seed and r['arm']==arm and r['facts']==32 and r['condition']=='untrained_in_vocab')}),flush=True)
                json_write(output/'progress.json',report)
    def scores(arm,condition):
        return [next(r['recall'] for r in report['cells'] if r['seed']==s and r['arm']==arm and r['facts']==32 and r['condition']==condition) for s in seeds]
    a=scores('clean','untrained_in_vocab');b=scores('augmentation','untrained_in_vocab');c=scores('invariance','untrained_in_vocab')
    delta=[x-y for x,y in zip(c,b)];regular=scores('invariance','regular');areg=scores('clean','regular')
    report['decision']={"primary_clean":a,"primary_augmentation":b,"primary_invariance":c,
                        "incremental_C_minus_B":delta,"primary_means":{"clean":statistics.mean(a),"augmentation":statistics.mean(b),"invariance":statistics.mean(c)},
                        "G1_incremental_invariance":statistics.mean(delta)>=.10 and sum(x>=0 for x in delta)>=4 and min(delta)>=-.05,
                        "G2_capability":statistics.mean(c)>=.8 and sum(x>=.7 for x in c)>=4 and statistics.mean(regular)>=.9 and statistics.mean(regular)-statistics.mean(areg)>=-.05,
                        "regular_invariance":regular,"OOV_invariance":scores('invariance','oov'),
                        "attention_replacement":"NOT TESTED","abstention":"descriptive diagnostics only, not a learned-ability gate"}
    report['total_wall_seconds']=time.perf_counter()-total_start
    report['frozen_manifest_sha256']=fingerprint(output/'frozen_manifest.json')
    for path,digest in manifest['files'].items():assert fingerprint(root/path)==digest,"source modified during run"
    json_write(output/'results.json',report)
    print(json.dumps(report['decision'],indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('evidence/run'))
    parser.add_argument('--seeds',default='61,62,63,64,65');parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--trials',type=int,default=64)
    args=parser.parse_args();run(Path(__file__).resolve().parent.parent,args.output,
                                tuple(map(int,args.seeds.split(','))),args.steps,args.trials)
