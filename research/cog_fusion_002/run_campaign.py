"""Lock and execute six declared fits. No held-data computation in this process."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import HERE, PARENT, SEEDS, ARMS, parent_checkpoint, sha256, write_json


def main():
    evidence = HERE/'evidence'
    lock_path = evidence/'TRAINING_LOCK.json'
    if lock_path.exists():
        raise FileExistsError('refusing to replace an existing campaign lock')
    # Refuse before touching locks or original outputs.
    outputs = [evidence/'runs'/f'{arm}_seed{seed}' for seed in SEEDS for arm in ARMS]
    if any(p.exists() for p in outputs):
        raise FileExistsError('campaign output already exists')
    data = evidence/'data/train.json'
    required = [HERE/'PROTOCOL.md',HERE/'common.py',HERE/'data.py',HERE/'train.py',
                HERE/'evaluate.py',HERE/'run_campaign.py',HERE/'runtime_eval.py',
                HERE/'preservation_eval.py',HERE/'state_interventions.py',
                evidence/'RUNTIME_PROTOCOL.json',evidence/'LANGUAGE_PROTOCOL.json',
                evidence/'data/DATA_MANIFEST.json',data,
                evidence/'PRETRAIN_REVIEW.json',
                PARENT/'fusion/model.py',PARENT/'fusion/engine.py',PARENT/'fusion/data.py',
                PARENT/'fusion/train.py',*sorted((PARENT/'fusion/native').glob('*.py'))]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'pretraining dependency missing: {path}')
    review = json.loads((evidence/'PRETRAIN_REVIEW.json').read_text())
    if not review.get('pass'):
        raise RuntimeError('pretraining engineering and data review did not pass')
    lock = {
        'schema':'cog-fusion-002-lock-v1','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_sha256':{str(p.relative_to(HERE.parent)):sha256(p) for p in required},
        'parent_sha256':{str(seed):sha256(parent_checkpoint(seed)) for seed in SEEDS},
        'seeds':SEEDS,'arms':ARMS,'updates':1200,'teacher_updates':100,'batch_families':4,
        'cpu_cap_seconds_per_fit':3600,'threads_per_fit':1,'parallel_fits':6,
        'native_lr':.000075,'interface_lr':.0005,'selected_checkpoint':'fixed_final_update',
        'held_evaluation_before_this_lock':False}
    write_json(lock_path,lock)
    logs = evidence/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    jobs=[]
    for seed in SEEDS:
        for arm in ARMS:
            name=f'{arm}_seed{seed}'
            output=evidence/'runs'/name
            handle=(logs/f'{name}.log').open('x')
            command=[sys.executable,str(HERE/'train.py'),'--parent',str(parent_checkpoint(seed)),
                     '--train-data',str(data),'--shakespeare',str(PARENT/'evidence/reproduction_inputs/train.bin'),
                     '--output',str(output),'--arm',arm,'--seed',str(seed),'--steps','1200',
                     '--teacher-steps','100','--pairs','4','--threads','1','--cpu-cap','3600',
                     '--native-lr','.000075','--lr','.0005']
            process=subprocess.Popen(command,cwd=HERE,env=env,stdout=handle,stderr=subprocess.STDOUT)
            jobs.append({'seed':seed,'arm':arm,'pid':process.pid,'process':process,'log':handle,'command':command})
    public=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs]
    write_json(evidence/'CAMPAIGN_JOBS.json',public)
    print(json.dumps(public),flush=True)
    while any(job['process'].poll() is None for job in jobs):
        time.sleep(5)
    results=[]
    for job in jobs:
        job['log'].close()
        results.append({'seed':job['seed'],'arm':job['arm'],'returncode':job['process'].returncode})
    write_json(evidence/'CAMPAIGN_EXIT.json',results)
    print(json.dumps(results),flush=True)
    if any(r['returncode'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
