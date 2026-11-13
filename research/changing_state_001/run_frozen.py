"""Frozen original-checkpoint development diagnostic; never fits weights."""
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from stress_data import generate, validate, encode, digest, VOCAB, REGIMES, DEPTHS
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2); f.write('\n')

def inputs(source):
    address=source/'experiments/csm_address_002'
    manifest=json.loads((address/'prefit_manifest.json').read_text())
    for entry in manifest['files']:
        path=address/entry['path']
        if sha(path)!=entry['sha256']: raise RuntimeError('Original source mismatch: '+str(path))
    panels=json.loads((address/'data/panels.json').read_text())
    forbidden={r['source_world_sha256'] for r in panels['train']}
    return address,forbidden

def prepare(source):
    address,forbidden=inputs(source)
    worlds=generate(forbidden)
    checks=validate(worlds,forbidden)
    write(ROOT/'worlds.json',worlds)
    files={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in ('.py','.md') or p.name=='worlds.json'}
    dependencies={str(p.relative_to(source)):sha(p) for p in [address/'data.py',address/'model_adapter.py',
        address/'prefit_manifest.json',source/'experiments/csm_foundation_001/model.py',
        source/'experiments/csm_foundation_001/baselines.py']}
    checkpoints={}
    for arm in ['raw','supervised','supplied_controller','transformer']:
        for seed in [101,202,303]:
            p=address/'runs'/f'{arm}_{seed}'/'checkpoint.pt'
            expected=json.loads((p.parent/'complete.json').read_text())['checkpoint_sha256']
            if sha(p)!=expected: raise RuntimeError('Checkpoint mismatch')
            checkpoints[f'{arm}_{seed}']=expected
    write(ROOT/'freeze.json',{'files':files,'dependencies':dependencies,'checkpoints':checkpoints,
        'checks':checks,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'status':'development diagnostic; no new fits; no protected TEST'})
    print(json.dumps(checks))

def summarize(records):
    families=defaultdict(list)
    for r in records: families[r['family']].append(r['correct'])
    changed=[r for r in records if r['changed']]
    untouched=[r for r in records if not r['changed']]
    def acc(xs):return sum(r['correct'] for r in xs)/len(xs) if xs else None
    return {'queries':len(records),'accuracy':acc(records),'worlds':len(families),
        'complete_world_accuracy':sum(all(v) for v in families.values())/len(families),
        'changed_queries':len(changed),'changed_accuracy':acc(changed),
        'stale_errors':sum(r['prediction']==r['prior'] for r in changed),
        'stale_rate':sum(r['prediction']==r['prior'] for r in changed)/len(changed) if changed else None,
        'untouched_queries':len(untouched),'untouched_accuracy':acc(untouched)}

def run(source):
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for name,h in frozen['files'].items():
        if sha(ROOT/name)!=h: raise RuntimeError('Frozen file changed: '+name)
    for name,h in frozen['dependencies'].items():
        if sha(source/name)!=h: raise RuntimeError('Frozen dependency changed: '+name)
    address,forbidden=inputs(source)
    worlds=json.loads((ROOT/'worlds.json').read_text()); validate(worlds,forbidden)
    output=ROOT/'results';output.mkdir(exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(address))
    adapter=importlib.import_module('model_adapter')
    historical_data=importlib.import_module('data')
    # Exact token/label parity with the unchanged original encoder before scoring.
    for w in worlds:
        for q in range(4):
            ref=historical_data.encode({'writes':w['writes'],'world_id':w['family'],'family_id':w['family']},q)
            row=encode(w,q)
            for k in ('input_ids','answer','destination_targets','edit_targets'):
                if row[k]!=ref[k]: raise RuntimeError('Encoder parity failure')
    summary=[]; start=time.perf_counter()
    receipt={'pid':os.getpid(),'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'device':'cpu','threads':1,'platform':platform.platform(),
        'freeze_sha256':sha(ROOT/'freeze.json'),'runtime_executable':sys.executable,
        'loaded_modules':{},'evaluation_kind':'instrumented batched independent query replay, not session timing'}
    for module in (adapter,adapter._reference,adapter._baseline,historical_data):
        receipt['loaded_modules'][module.__name__]={'path':module.__file__,'sha256':sha(module.__file__)}
    with (output/'predictions.jsonl').open('x') as log, torch.inference_mode():
        for arm in ['supervised','raw','supplied_controller','transformer']:
            for seed in [101,202,303]:
                path=address/'runs'/f'{arm}_{seed}'/'checkpoint.pt'
                if sha(path)!=frozen['checkpoints'][f'{arm}_{seed}']: raise RuntimeError('Changed checkpoint')
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                model=adapter.build_model(arm,seed,len(VOCAB));model.load_state_dict(checkpoint['model']);model.eval()
                original={k:v.clone() for k,v in model.state_dict().items()}
                for regime in REGIMES:
                    for depth in DEPTHS:
                        rows=[encode(w,q) for w in worlds if w['regime']==regime and w['updates']==depth for q in range(4)]
                        length=len(rows[0]['input_ids'])
                        if arm=='transformer' and length>128:
                            summary.append({'arm':arm,'seed':seed,'regime':regime,'updates':depth,'status':'NOT_SCORED','reason':'outside bounded <=128 token diagnostic; not a failed prediction'})
                            continue
                        records=[]; routes={'destination_correct':0,'destination_total':0,'false_writes':0,'nonwrite_positions':0,'missed_writes':0,'write_positions':0}
                        tick=time.perf_counter()
                        for offset in range(0,len(rows),64):
                            group=rows[offset:offset+64]; batch=adapter.tensors(group)
                            # No target-bearing batch object enters learned forward.
                            if arm=='transformer': logits=model(batch['ids'])[:,-1]; cap={}
                            else:
                                supplied=(batch['dest'],batch['edit']) if arm=='supplied_controller' else None
                                logits,cap=model.sequence(batch['ids'],supplied_control=supplied,collect=True)
                            if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite logits')
                            predictions=logits.argmax(-1).tolist()
                            for row,pred in zip(group,predictions):
                                r={k:row[k] for k in ('family','block','regime','updates','query','answer','prior','changed')}
                                r.update(arm=arm,seed=seed,prediction=pred,correct=pred==row['answer'])
                                log.write(json.dumps(r,sort_keys=True)+'\n');records.append(r)
                            if cap and arm!='supplied_controller':
                                mask=batch['dest']>=0; ed=cap['value_edit'].argmax(-1); truth=batch['edit'].bool()
                                routes['destination_correct']+=int(((cap['destination'].argmax(-1)==batch['dest'])&mask).sum())
                                routes['destination_total']+=int(mask.sum())
                                routes['false_writes']+=int(((ed==1)&~truth).sum());routes['nonwrite_positions']+=int((~truth).sum())
                                routes['missed_writes']+=int(((ed==0)&truth).sum());routes['write_positions']+=int(truth.sum())
                        result={'arm':arm,'seed':seed,'regime':regime,'updates':depth,'status':'SCORED','tokens':length,
                            **summarize(records),'controller':routes if arm in ('raw','supervised') else None,
                            'instrumented_elapsed_s':time.perf_counter()-tick}
                        summary.append(result); print(arm,seed,regime,depth,round(result['accuracy'],4),flush=True)
                if any(not torch.equal(v,original[k]) for k,v in model.state_dict().items()): raise RuntimeError('Weights changed')
                if sha(path)!=frozen['checkpoints'][f'{arm}_{seed}']: raise RuntimeError('Checkpoint bytes changed')
    controls=[]
    for regime in REGIMES:
        for depth in DEPTHS:
            selected=[w for w in worlds if (w['regime'],w['updates'])==(regime,depth)]
            for policy in ('current_value_oracle','initial_value','last_global_value'):
                records=[]
                for w in selected:
                    for q in range(4):
                        r=encode(w,q)
                        val=dict(w['writes'])[q] if policy=='current_value_oracle' else dict(w['writes'][:4])[q] if policy=='initial_value' else w['writes'][-1][1]
                        pred=VOCAB[f'v{val}'];r.update(prediction=pred,correct=pred==r['answer']);records.append(r)
                controls.append({'policy':policy,'regime':regime,'updates':depth,**summarize(records)})
    receipt.update(elapsed_s=time.perf_counter()-start,weights_unchanged=True,
        predictions_sha256=sha(output/'predictions.jsonl'))
    write(output/'summary.json',summary);write(output/'controls.json',controls);write(output/'receipt.json',receipt)
    print('COMPLETE',receipt['elapsed_s'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--source',type=Path,required=True)
    args=p.parse_args(); (prepare if args.mode=='prepare' else run)(args.source.resolve())
