"""Use the Veritas Research Lab OS controller and its TrainLab worker."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from research_lab.core import Controller
from research_lab.evidence import put_node
from research_lab.execution import run_worker
from trainlab.runner import verify_run
HERE=Path(__file__).resolve().parent
PRIOR=Path(os.getenv('COG_PRIOR_ROOT','/workspace/delta-authority-048'))
PRIVATE=Path(os.getenv('COG_RUN_DIR','/workspace/cog-next-private'))
REC=PRIVATE/'recipe'
REC.mkdir(parents=True,exist_ok=True)
files=['execute.py','grounding.py','system_tests.py','panel.py','PROTOCOL.md']
for f in files: shutil.copyfile(HERE/f,REC/f)
deps=list((PRIOR/'recipe').rglob('*.py'))+list((PRIOR/'recipe/original_lrc').glob('*.pt'))+list((PRIOR/'recipe/original_lrc').glob('*.json'))+list((PRIOR/'recipe/original_csm').glob('*.npz'))+list((PRIOR/'portable').glob('*.npz'))+[PRIOR/'module-manifest.json']
for m in json.loads((PRIOR/'module-manifest.json').read_text()):
    if m['arm']=='candidate':
        folder=PRIOR/'ecosystem/research_lab/runs'/m['run_id']/'output'
        deps.extend(folder/f for f in m['checkpoints'])
pins={str(f.relative_to(PRIOR)):hashlib.sha256(f.read_bytes()).hexdigest() for f in deps}
(REC/'prior-inputs.json').write_text(json.dumps(pins,indent=2))
shutil.copyfile(REC/'prior-inputs.json',HERE/'prior-inputs.json')
lab=Controller(PRIVATE/'lab')
lab.program('cog-acquisition','Grounded acquisition and traced execution',18000,0)
lab.resource('cpu','cpu',1,0)
seeds=[5101,5107,5113]
for arm in ('candidate','control'):
    cfg=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',name='acquisition-'+arm,description='Grounded experience and qualified module trace',command=[sys.executable,'execute.py',arm],seed=seeds[0],timeout_seconds=600,source_kind='CONSTRUCTED',split_identity='cog-051-060-r1',inputs=files+['prior-inputs.json'],artifacts=['result.json','stable.sha256','csm.npz','roles.joblib','transformer.pt','gru.pt'],metrics={'success':'max'},budget={'cpu':1})
    (REC/(arm+'.json')).write_text(json.dumps(cfg))
p=dict(schema='research_lab.protocol.v1',id='acquisition',program='cog-acquisition',author='codex',question='Does a corrected learned route support novel grounded execution?',hypothesis='Correct routing improves end-state and trace agreement.',alternatives=['Language transfer may fail.','Changes can corrupt a persistent skill.'],predictions=['Correct routing helps.','Unobserved changes impair reuse.'],controls=['Same seeds and sensory experience.'],ablations=['Rotate the LRC result.'],metric='success',direction='max',minimum_effect=0,seeds=seeds,resource='cpu',slots=1,max_attempts=1,stopping_rule='Run all seeds and replay all outputs named for verification.',scope='ENGINEERING_ONLY',priority=1,source_commit='eec439492407b0aa26922c23b5ae40d95289f4eb',next_test='Independent authentic closed-loop replication',limitations=['Constructed control interface.'],candidate='candidate.json',control='control.json',replication_tolerance=0,replay_artifacts=['stable.sha256'])
lab.register(p,REC);put_node(lab,'acquisition.protocol','experiment',p)
for role in ('experiment','replication'):
    lab.worker('acquisition-'+role,role,'cpu')
    run_worker(lab,'acquisition-'+role)
    (PRIVATE/'status.json').write_text(json.dumps(lab.status(),indent=2))
for task in lab.status()['tasks']:
    if task['role']=='review': continue
    assert task['status']=='succeeded',task
    if task['role']=='replication':
        assert task['result']['all_match']
        (HERE/'replication.json').write_text(json.dumps(task['result'],indent=2))
        continue
    path=PRIVATE/'lab/runs'/task['result']['run_id']
    assert verify_run(path)['valid']
    run=json.loads((path/'run.json').read_text())
    dest=HERE/'results'/str(run['seed'])/run['name']
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path/'output/result.json',dest/'result.json')
    (dest/'receipt.json').write_text(json.dumps({k:run[k] for k in ('id','seed','inputs','artifacts','elapsed_seconds','metrics')},indent=2))
