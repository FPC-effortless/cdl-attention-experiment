"""Freeze exact COG-COMM-ROLE-001 replay actor(s); no prediction on invocation."""
from pathlib import Path
from hashlib import sha256
import json, shutil
import sys
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
IDs={1171:'20261122T160245-1e36aaf3461b',1271:'20261122T160254-beac69309987',1371:'20261122T160302-cf5576c5cd00'}
RECIPE_PINS={1171:'c58043f8a565369ffd6ad221e576059e393111178a883d4f1b48a855c685d11e',
1271:'7ae79c2fe99d6b7f1f622e7bbb499789dd47f90849606af27b83eb45607d93b4',
1371:'55b2524dd629af546ed688a250260a3d45f3953f35ce7916cd894549ff2f9556'}
WRAP="""import json, os, sys
from pathlib import Path
import torch
from cog_comm.role_actor import predict
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert sys.version_info[:2] == (3,12) and torch.__version__ == '2.10.0+cpu'
out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
_, actor_sha, cp_sha, seed, step = sys.argv
p=predict(Path('actor.json'),actor_sha,Path('checkpoint.pt'),cp_sha)
if (p['seed'],p['updates'],p['arm'],p['tokenizer_words'])!=(int(seed),int(step),'gated32',json.load(open('vocabulary.json'))['words']):
    raise ValueError('checkpoint identity or exact vocabulary changed')
(out/'predictions.json').write_text(json.dumps(p,allow_nan=False,separators=(',',':'))+'\\n')
(out/'metrics.json').write_text(json.dumps({'prediction_rows':len(p['logits'])})+'\\n')
"""
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,d):p.write_text(json.dumps(d,sort_keys=True,allow_nan=False,indent=2)+'\n')

def main(dest):
    dest.mkdir()
    pins=read(ROOT/'role-data-audit/audit.json')['input_hashes']
    ledger={}
    for seed, uid in IDs.items():
        run=ROOT/'fits-gated32'/uid
        rv=verify_run(run)
        assert rv['valid'],rv
        rr=read(run/'run.json')
        assert rr['seed']==seed and rr['id']==uid and rr['research_adequacy']=='EXECUTED_CRITERIA_MET'
        assert h(run/'experiment.json')==rr['files']['experiment.json']
        assert h(ROOT/f'admissions/gated32-{seed}/recipe.json')==RECIPE_PINS[seed]
        assert read(run/'experiment.json')==read(ROOT/f'admissions/gated32-{seed}/recipe.json')
        assert h(run/'output'/'vocabulary.json')==rr['artifacts']['vocabulary.json']
        for step in (128,1280):
          for split in ('fit','control'):
            case=dest/f'gated32-{seed}-{step}-{split}'
            case.mkdir()
            files=['cog_comm/'+n+'.py' for n in ('__init__','model','mix','router','objective','role_mixers','role_actor','role_judge','role_control')]+['nanocog/__init__.py','nanocog/model.py']
            for rel in files:
                p=case/rel;p.parent.mkdir(exist_ok=True)
                shutil.copyfile(ROOT/'smoke'/rel,p)
                if rel in rr['inputs']:
                    assert h(p)==rr['inputs'][rel]
            actor=ROOT/'panels'/f'{split}_actor.json'
            assert h(actor)==pins[actor.name]
            shutil.copyfile(actor,case/'actor.json')
            weights=run/'output'/f'checkpoint-{step}.pt'
            assert h(weights)==rr['artifacts'][weights.name]
            shutil.copyfile(weights,case/'checkpoint.pt')
            shutil.copyfile(run/'output'/'vocabulary.json',case/'vocabulary.json')
            (case/'observe_actor.py').write_text(WRAP)
            files+=['actor.json','checkpoint.pt','vocabulary.json','observe_actor.py']
            recipe=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
                name=case.name,description='Exact read-only trained checkpoint replay; public synthetic prompts only.',
                seed=seed,timeout_seconds=30,source_kind='CONSTRUCTED',split_identity=f'ROLE-001-fresh-{split}-actor',
                command=['python','observe_actor.py',h(actor),h(weights),str(seed),str(step)],inputs=files,
                artifacts=['predictions.json'],metrics={'prediction_rows':'max'},
                budget={'prediction_batches':320 if split=='fit' else 120})
            write(case/'recipe.json',recipe)
            ledger[case.name]=dict(recipe_sha256=h(case/'recipe.json'),inputs={f:h(case/f) for f in files},
                                  source_run=uid,actor_sha256=h(actor),checkpoint_sha256=h(weights),
                                  seed=seed,step=step,split=split)
    write(dest/'ledger.json',ledger)
if __name__=='__main__': main(Path(sys.argv[1]))
