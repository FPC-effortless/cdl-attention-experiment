"""Prepare only independent, exact confirmation predictions, no judging."""
from pathlib import Path
from hashlib import sha256
import json, shutil
from trainlab.runner import verify_run
BASE=Path('/workspace/shared/cog-role-001')
PAIR_BASE=Path('/workspace/shared/cog-role-002')
PIN_ACTOR='536ba918ab5122549cd1142df0c02789fb406731a2fdf4928db75ac04a1c1555'
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
        old_fit=PAIR_BASE/'fits'/read(PAIR_BASE/f'admissions/gated32-{s}/cli-output.json')['id']
        fit=Path('fits')/read(Path(f'admissions/gated32-{s}/cli-output.json'))['id']
        for label,run in [('clip1',old_fit),('clip2',fit)]:
            rr=read(run/'run.json'); obs=read(run/'output/fit.json')
            req(verify_run(run)['valid'] and rr['status']=='succeeded' and
                rr['research_adequacy']=='EXECUTED_CRITERIA_MET' and rr['seed']==s,
                'invalid fit predecessor')
            expected=1.0 if label=='clip1' else 2.0
            req(obs['plan']['clip']==expected and obs['plan']['prefix_ce_weight']==.25 and
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
            recipe.update(name=d.name, description='ROLE-003 frozen local confirmation actor only',
                          split_identity='ROLE-003-fresh-synthetic-confirmation-actor')
            recipe['command'][2:4]=[PIN_ACTOR,sha(ck)]
            write(d/'recipe.json',recipe)
            ledger[d.name]=dict(actor_sha256=PIN_ACTOR,checkpoint_sha256=sha(ck),seed=s,endpoint=n,
                                arm=label,fit_id=rr['id'],fit_run_sha=sha(run/'run.json'),
                                recipe_sha256=sha(d/'recipe.json'),inputs={x:sha(d/x) for x in recipe['inputs']})
    write(root/'ledger.json',ledger)
    print(sha(root/'ledger.json'))
if __name__=='__main__': main()
