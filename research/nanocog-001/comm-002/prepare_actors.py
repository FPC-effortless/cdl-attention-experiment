"""Prepare only independent, exact confirmation predictions, no judging."""
from pathlib import Path
from hashlib import sha256
import json, shutil
from trainlab.runner import verify_run
BASE=Path('/workspace/shared/cog-role-001')
PIN_ACTOR='201a63921dfaaf601b03e03e7dc80858e9958b71fe0eef8e6b8572d8d459aac1'
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,d):p.write_text(json.dumps(d,sort_keys=True,allow_nan=False,indent=2)+'\n')
def req(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    root=Path('actor-admissions')
    root.mkdir()
    ap=Path('panel/confirm_actor.json')
    req(sha(ap)==PIN_ACTOR,'wrong fresh actor pin')
    ledger={}
    for s in (1171,1271,1371):
      for n in (128,1280):
        old=BASE/'actor-admissions'/f'gated32-{s}-{n}-control'
        old_ref=read(BASE/'actor-admissions/ledger.json')[old.name]
        old_fit=BASE/'fits-gated32'/old_ref['source_run']
        fit=Path('fits')/read(Path(f'admissions/gated32-{s}/cli-output.json'))['id']
        for label,run in [('w1',old_fit),('w025',fit)]:
            rr=read(run/'run.json'); obs=read(run/'output/fit.json')
            req(verify_run(run)['valid'] and rr['status']=='succeeded' and
                rr['research_adequacy']=='EXECUTED_CRITERIA_MET' and rr['seed']==s,
                'invalid fit predecessor')
            expected=1.0 if label=='w1' else .25
            req(obs['plan'].get('prefix_ce_weight',1.0)==expected and
                obs['plan']['arm']=='gated32','wrong objective lineage')
            d=root/f'{label}-{s}-{n}';d.mkdir()
            recipe=read(old/'recipe.json')
            source=[x for x in recipe['inputs'] if x not in ('actor.json','checkpoint.pt','vocabulary.json')]
            for x in source:
                req(sha(old/x)==old_ref['inputs'][x], 'old source hash mismatch')
                dest=d/x;dest.parent.mkdir(exist_ok=True);shutil.copyfile(old/x,dest)
            shutil.copyfile(ap,d/'actor.json')
            ck=run/'output'/f'checkpoint-{n}.pt';v=run/'output/vocabulary.json'
            for p,a in [(ck,'checkpoint.pt'),(v,'vocabulary.json')]:
                req(sha(p)==rr['artifacts'][p.name],'fit artifact mismatch')
                shutil.copyfile(p,d/a)
            recipe.update(name=d.name, description='ROLE-002 frozen local confirmation actor only',
                          split_identity='ROLE-002-fresh-synthetic-confirmation-actor')
            recipe['command'][2:4]=[PIN_ACTOR,sha(ck)]
            write(d/'recipe.json',recipe)
            ledger[d.name]=dict(actor_sha256=PIN_ACTOR,checkpoint_sha256=sha(ck),seed=s,endpoint=n,
                                arm=label,fit_id=rr['id'],fit_run_sha=sha(run/'run.json'),
                                recipe_sha256=sha(d/'recipe.json'),inputs={x:sha(d/x) for x in recipe['inputs']})
    write(root/'ledger.json',ledger)
    print(sha(root/'ledger.json'))
if __name__=='__main__': main()
