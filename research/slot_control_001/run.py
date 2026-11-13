"""Three closed fixed-budget control fits, then frozen development evaluation."""
import argparse, hashlib, json, os, platform, sys, time, random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STRESS=ROOT.parent/'changing_state_001'
sys.path.insert(0,str(STRESS))
from run_frozen import inputs,sha,write,summarize
from stress_data import encode,REGIMES,DEPTHS,VOCAB

def prepare(source):
    address,forbidden=inputs(source)
    files={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in ('.py','.md')}
    write(ROOT/'freeze.json',{'files':files,'stress_freeze_sha256':sha(STRESS/'freeze.json'),
        'training_data_sha256':sha(address/'data/panels.json'),
        'historical_config_sha256':sha(address/'config.json'),
        'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seeds':[101,202,303]})

def verify(source):
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for name,h in frozen['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Changed control source')
    if sha(STRESS/'freeze.json')!=frozen['stress_freeze_sha256']:raise RuntimeError('Changed stress freeze')
    stress=json.loads((STRESS/'freeze.json').read_text())
    for name,h in stress['files'].items():
        if sha(STRESS/name)!=h:raise RuntimeError('Changed stress input')
    address,_=inputs(source)
    if sha(address/'data/panels.json')!=frozen['training_data_sha256']:raise RuntimeError('Changed training data')
    if sha(address/'config.json')!=frozen['historical_config_sha256']:raise RuntimeError('Changed historical configuration')
    return address

def setup(address):
    import torch
    import numpy as np
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sys.path.insert(0,str(address))
    import model_adapter as adapter
    from model import SlotStore
    return torch,np,adapter,SlotStore

def train(source):
    address=verify(source)
    if (STRESS/'results').exists():raise RuntimeError('Stress scoring already opened')
    runs=ROOT/'runs';runs.mkdir(exist_ok=False)
    torch,np,adapter,SlotStore=setup(address)
    cfg=json.loads((address/'config.json').read_text())
    examples=json.loads((address/'data/panels.json').read_text())['train'];bank=adapter.tensors(examples)
    receipts=[]
    for seed in (101,202,303):
        directory=runs/str(seed);directory.mkdir()
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=SlotStore();model.train()
        generator=torch.Generator().manual_seed(seed+900000)
        schedule=torch.randint(len(examples),(256,32),generator=generator)
        schedule_hash=hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
        original=json.loads((address/'runs'/f'supervised_{seed}'/'complete.json').read_text())
        if schedule_hash!=original['schedule_sha256']:raise RuntimeError('Unmatched schedule')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        start=time.perf_counter()
        with (directory/'updates.jsonl').open('x') as log:
            for step,index in enumerate(schedule,1):
                batch={k:v[index] for k,v in bank.items()};optimizer.zero_grad(set_to_none=True)
                logits,cap=model.sequence(batch['ids'],collect=True)
                loss,parts=adapter.objective('supervised',logits,cap,batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                entry={'step':step,'loss':float(loss.detach()),'parts':parts,'gradient_norm':float(norm),
                    'train_batch_accuracy':float((logits.argmax(-1)==batch['answers']).float().mean())}
                log.write(json.dumps(entry)+'\n');log.flush()
                if step%64==0:print(seed,step,round(entry['train_batch_accuracy'],4),flush=True)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'seed':seed,'updates':256,
                    'rng_torch':torch.get_rng_state()},directory/'checkpoint.pt')
        receipt={'seed':seed,'pid':os.getpid(),'python':platform.python_version(),'torch':torch.__version__,
            'numpy':np.__version__,'threads':1,'device':'cpu','parameters':sum(p.numel() for p in model.parameters()),
            'updates':256,'answer_exposures':8192,'schedule_sha256':schedule_hash,
            'checkpoint_sha256':sha(directory/'checkpoint.pt'),'training_log_sha256':sha(directory/'updates.jsonl'),
            'freeze_sha256':sha(ROOT/'freeze.json'),'elapsed_s':time.perf_counter()-start}
        write(directory/'complete.json',receipt);receipts.append(receipt)
    write(ROOT/'training_closed.json',{'runs':receipts,'all_three_closed_before_stress_scoring':not (STRESS/'results').exists()})

def validate_closure(source):
    address=verify(source)
    closure=json.loads((ROOT/'training_closed.json').read_text())
    receipts=closure.get('runs',[])
    if len(receipts)!=3 or sorted(r.get('seed',-1) for r in receipts)!=[101,202,303]:
        raise RuntimeError('Closure must contain exactly seeds 101/202/303')
    if closure.get('all_three_closed_before_stress_scoring') is not True:
        raise RuntimeError('Fits did not close before scoring')
    for receipt in receipts:
        directory=ROOT/'runs'/str(receipt['seed'])
        if receipt!=json.loads((directory/'complete.json').read_text()):
            raise RuntimeError('Closure receipt differs from completed run')
        if receipt['updates']!=256 or receipt['answer_exposures']!=8192:
            raise RuntimeError('Wrong training budget')
        if receipt['freeze_sha256']!=sha(ROOT/'freeze.json'):
            raise RuntimeError('Training used different freeze')
        if sha(directory/'checkpoint.pt')!=receipt['checkpoint_sha256']:
            raise RuntimeError('Checkpoint bytes changed')
        if sha(directory/'updates.jsonl')!=receipt['training_log_sha256']:
            raise RuntimeError('Training log bytes changed')
        entries=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        if [e['step'] for e in entries]!=list(range(1,257)):
            raise RuntimeError('Incomplete or reordered training steps')
        historical=json.loads((address/'runs'/f"supervised_{receipt['seed']}"/'complete.json').read_text())
        if receipt['schedule_sha256']!=historical['schedule_sha256']:
            raise RuntimeError('Training schedules differ')
    return closure

def load_model(seed,torch,SlotStore):
    path=ROOT/'runs'/str(seed)/'checkpoint.pt'
    receipt=json.loads((path.parent/'complete.json').read_text())
    if sha(path)!=receipt['checkpoint_sha256']:raise RuntimeError('Changed control checkpoint')
    model=SlotStore();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model.eval()
    return model

def evaluate(source):
    address=verify(source);closure=validate_closure(source)
    torch,np,adapter,SlotStore=setup(address)
    worlds=json.loads((STRESS/'worlds.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=False)
    summaries=[]
    with (out/'predictions.jsonl').open('x') as log,torch.inference_mode():
        for seed in (101,202,303):
            model=load_model(seed,torch,SlotStore);old={k:v.clone() for k,v in model.state_dict().items()}
            for regime in REGIMES:
                for depth in DEPTHS:
                    rows=[encode(w,q) for w in worlds if (w['regime'],w['updates'])==(regime,depth) for q in range(4)];records=[]
                    for offset in range(0,len(rows),64):
                        group=rows[offset:offset+64];batch=adapter.tensors(group)
                        logits,_=model.sequence(batch['ids'])
                        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite')
                        for row,pred in zip(group,logits.argmax(-1).tolist()):
                            r={k:row[k] for k in ('family','block','regime','updates','query','answer','prior','changed')}
                            r.update(arm='slot_store',seed=seed,prediction=pred,correct=pred==row['answer'])
                            records.append(r);log.write(json.dumps(r,sort_keys=True)+'\n')
                    summary={'arm':'slot_store','seed':seed,'regime':regime,'updates':depth,**summarize(records)}
                    summaries.append(summary);print(seed,regime,depth,summary['accuracy'],flush=True)
            if any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()):raise RuntimeError('Changed weights')
    write(out/'summary.json',summaries);write(out/'receipt.json',{'pid':os.getpid(),'weights_unchanged':True,
        'predictions_sha256':sha(out/'predictions.jsonl'),'closure_sha256':sha(ROOT/'training_closed.json'),
        'freeze_sha256':sha(ROOT/'freeze.json'),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','evaluate']);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'train':train,'evaluate':evaluate}[a.mode](a.source.resolve())
