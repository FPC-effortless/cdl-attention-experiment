"""Prepare read-only actor closures for the exact nine comparator trajectories."""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import sys
from trainlab.runner import verify_run
from prep_replay import WRAP, ROOT, read, h, write
BASE=ROOT/'admissions-comparators'
ARMS=('static32','attention32','gated48')
SEEDS=(1171,1271,1371)
DEST=ROOT/'actor-admissions-comparators'
def require(ok):
    if not ok: raise ValueError('unexpected comparator input bytes/identity')
def main():
    DEST.mkdir(exist_ok=False)
    ledger={}
    for arm in ARMS:
      for seed in SEEDS:
        case=BASE/f'{arm}-{seed}'
        r=read(case/'cli-output.json')
        run=ROOT/'fits-comparators'/r['id']
        require(verify_run(run)['valid'] and r==read(run/'run.json') and r['seed']==seed and
                r['research_adequacy']=='EXECUTED_CRITERIA_MET')
        require(read(run/'experiment.json')==read(case/'recipe.json'))
        require(read(run/'output/fit.json')['plan']['arm']==arm)
        for step in (128,1280):
          for split in ('fit','control'):
            # Reuse the exact actor skeleton and authoritative panel/vocabulary
            # projections, but replace checkpoint, arm/seed/step and output pin.
            source=ROOT/'actor-admissions'/f'gated32-{seed}-{step}-{split}'
            d=DEST/f'{arm}-{seed}-{step}-{split}'
            shutil.copytree(source,d,ignore=shutil.ignore_patterns('cli-output.json','__pycache__'))
            recipe=read(d/'recipe.json')
            for k in recipe['inputs']:
                if k in r['inputs']:
                    require(r['inputs'][k]==h(d/k))
            for f,z in [('checkpoint.pt',f'checkpoint-{step}.pt'),('vocabulary.json','vocabulary.json')]:
                require(h(run/'output'/z)==r['artifacts'][z]); shutil.copyfile(run/'output'/z,d/f)
            (d/'observe_actor.py').write_text(WRAP.replace("'gated32',",f"'{arm}',"))
            recipe['name']=d.name
            recipe['command'][-3]=h(d/'checkpoint.pt')
            write(d/'recipe.json',recipe)
            ledger[d.name]=dict(recipe_sha256=h(d/'recipe.json'),inputs={f:h(d/f) for f in recipe['inputs']},
                 source_run=r['id'],actor_sha256=h(d/'actor.json'),checkpoint_sha256=h(d/'checkpoint.pt'),
                 seed=seed,step=step,split=split,arm=arm)
    write(DEST/'ledger.json',ledger)
if __name__=='__main__':main()
