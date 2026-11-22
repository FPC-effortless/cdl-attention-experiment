"""Materialize byte-bound geometry actor recipes; no subprocess or model import."""
import json,shutil
from hashlib import sha256
from pathlib import Path
from trainlab.runner import verify_run,load_config

P=Path(__file__).resolve().parent
OLD=P.parent/'cog-role-diag'
FITS=P.parent/'cog-role-002/fits'
OLD_LEDGER='122dbb6f4417a39d1acf32e970b45ae8722657e7b09dc54c375ea9aa809ac612'
OLD_SEAL='430460c8260d09537940a1c5efbd034df411baf3f0d801400121e9d67483d766'
SEEDS=(1171,1371)

def sh(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
    if pin is not None and sh(p)!=pin:raise ValueError('unexpected hash '+str(p))
    return json.loads(p.read_bytes())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def check(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    src=OLD/'probe_admissions'; old=read(src/'ledger.json',OLD_LEDGER)
    seal=read(OLD/'PROBE_SEAL.json',OLD_SEAL)
    new=P/'geometry_admissions_v2';new.mkdir()
    led={}
    for seed in SEEDS:
        k=f'probe-{seed}'; case=old[k]; fit=FITS/case['fit_id']; pre=OLD/'probe_runs'/seal[k]['id']
        check(verify_run(fit)['valid'] and sh(fit/'run.json')==case['fit_run_sha'],'old FIT proof')
        check(verify_run(pre)['valid'] and sh(pre/'run.json')==seal[k]['receipt_sha256'],'old probe proof')
        check(case['checkpoint_sha256']==case['inputs']['checkpoint.pt']==read(fit/'run.json')['artifacts']['checkpoint-1280.pt'],
              'old weight identity')
        d=new/f'geometry-{seed}';d.mkdir()
        files={n:src/k/n for n in case['inputs'] if n in ('actor.json','checkpoint.pt') or n.startswith(('cog_comm/','nanocog/'))}
        check(all(sh(p)==case['inputs'][n] for n,p in files.items()),'old worker input changed')
        for n in ('probability_dose.py','dose_summary.py','run_geometry.py','GATE_PROBE_002_PROTOCOL_V2.md'):
            files[n]=P/n
        for n,p in files.items():
            o=d/n;o.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,o)
        x=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
               name=d.name,description='Natural query-gate geometry only, no interventions',
               command=['python','run_geometry.py',case['actor_sha256'],case['checkpoint_sha256'],str(seed)],
               seed=seed,timeout_seconds=45,source_kind='CONSTRUCTED',
               split_identity='ROLE-GATE-001-reused-exposed-synthetic-actor',inputs=sorted(files),
               artifacts=['geometry.json'],metrics=dict(probability_rows='max',model_forwards='min',optimizer_updates='min',
               intervention_forwards='min'),budget=dict(inference_batches=32))
        write(d/'recipe.json',x);check(load_config(d/'recipe.json')==x,'native recipe rejected/changed')
        led[d.name]=dict(seed=seed,recipe_sha256=sh(d/'recipe.json'),actor_sha256=case['actor_sha256'],
                         checkpoint_sha256=case['checkpoint_sha256'],role002_fit_run_sha=case['fit_run_sha'],
                         previous_probe_run_sha=seal[k]['receipt_sha256'],inputs={n:sh(d/n) for n in files})
    write(new/'ledger.json',led)
    print(sh(new/'ledger.json'))
if __name__=='__main__':main()
