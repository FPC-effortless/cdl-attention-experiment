import json
import shutil
import sys
from pathlib import Path
import numpy as np
from trainlab.runner import run_experiment, verify_run

HERE=Path(__file__).resolve().parent
data=Path('/workspace')
old=data/'issue-history-study/recipe'
new=data/'issue-history-followup/recipe'
work=data/'cog-next-private/authentic'
work.mkdir(parents=True,exist_ok=True)
for name in ('models.py','evaluation.py','design.json','state_store.py','wr006.py'):
    shutil.copyfile(new/name,work/name)
shutil.copyfile(HERE/'authentic.py',work/'authentic.py')
arch=json.loads((old/'dataset.json').read_text())
ids=[i for i,r in enumerate(arch) if r['split'] in ('train','validation')]
rows=[arch[i] for i in ids]+json.loads((new/'dataset.json').read_text())
assert not ({r['issue'] for r in rows if r['split'] in ('train','validation')} &
            {r['issue'] for r in rows if r['split'] in ('test','external')})
(work/'combined.json').write_text(json.dumps(rows))
np.save(work/'combined.npy',np.concatenate([np.load(old/'semantic.npy')[ids],np.load(new/'semantic.npy')]))
for seed in (67003,67009,67021):
    cfg=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',name='fresh-grounding',
             description='Fresh issue strict, pooled and separate consequence supervision',command=[sys.executable,'authentic.py'],
             seed=seed,timeout_seconds=600,source_kind='AUTHENTIC_CANDIDATE',split_identity='issue-holdout-jul-sep-2026',
             inputs=['authentic.py','combined.json','combined.npy','models.py','evaluation.py','design.json','state_store.py','wr006.py'],
             artifacts=['issue-results.json','issue-predictions.json','issue-checkpoint.joblib'],metrics={'ap':'max'},budget={'cpu':1})
    path=work/f'{seed}.json'
    path.write_text(json.dumps(cfg))
    run=run_experiment(path,work/'runs')
    assert run['status']=='succeeded' and verify_run(work/'runs'/run['id'])['valid'],run['id']
    dest=HERE/'results'/str(seed)
    dest.mkdir(parents=True,exist_ok=True)
    for f in ('issue-results.json','issue-predictions.json'):
        shutil.copyfile(work/'runs'/run['id']/'output'/f,dest/f)
    (dest/'authentic-receipt.json').write_text(json.dumps({k:run[k] for k in ('id','inputs','artifacts','metrics','elapsed_seconds')}))
