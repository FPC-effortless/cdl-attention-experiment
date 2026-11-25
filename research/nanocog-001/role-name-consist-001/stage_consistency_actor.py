"""Prepare exact 13 no-update actor jobs after reviewed FIT only."""
import json,shutil
from hashlib import sha256
from pathlib import Path
from trainlab.runner import verify_run
from score_consistency import AS,roster
R=Path(__file__).resolve().parent

BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
ARMS=('consistency000','consistency025');SEEDS=(1771,1871,1971)
def read(p):return json.loads(p.read_bytes())
def h(p):return sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')
def need(ok,why):
    if not ok:raise ValueError(why)
def main(fits,registry,fit_runs):
    intake=read(registry); parent=read(fits/'ledger.json')
    need(set(parent)==set(intake['runs'])=={f'{a}-{s}' for a in ARMS for s in SEEDS},'fit roster')
    out=R/'consist_actor_admissions';out.mkdir(exist_ok=False)
    roster={}; format_source=next(iter(parent))
    def stage(arm,seed,step,fitcase,checkp,weight_pin,split):
        A=AS[split]; count=960 if split=='ordinary' else 768
        src=fits/fitcase
        entry=parent[fitcase]
        d=out/f'{arm}-{seed}-{step}-{split}';d.mkdir()
        # No train.json or adapter supervision in any actor.
        files=[x for x in entry['inputs'] if x.startswith(('cog_comm/','nanocog/')) or x in
               ('tap_gate_residual.py','frozen_gate_adapter.py','fit_adapter.py','checkpoint.pt')]
        for f in files:
            need(h(src/f)==entry['inputs'][f],'predecessor source mismatch')
            p=d/f;p.parent.mkdir(exist_ok=True);shutil.copyfile(src/f,p)
        need(h(d/'checkpoint.pt')==BASE and h(R/f'eval_panels/{split}_actor.json')==A,'actor or base changed')
        shutil.copyfile(R/f'eval_panels/{split}_actor.json',d/'actor.json');shutil.copyfile(R/'predict_consistency.py',d/'predict_consistency.py')
        predecessor=R.parent/'cog-role-repair-004/fit_admissions_v2/tap_compare-1471/fit_adapter.py'
        need(h(predecessor)=='59335cda6b5b4bece8a988bcc6d5c68e798f70869a6d64f994af416418d73cf7','pinned base loader source')
        shutil.copyfile(predecessor,d/'fit_adapter.py')
        files+=['actor.json','predict_consistency.py','fit_adapter.py']
        if checkp is not None:
            need(h(checkp)==weight_pin,'exact adapter digest')
            shutil.copyfile(checkp,d/'adapter.pt');files.append('adapter.pt')
        r=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
            name=d.name,description='Exact frozen/new adapter replay on prospective public synthetic actor.',
            command=['python','predict_consistency.py',arm,str(seed),str(step),weight_pin,split],
            seed=seed,timeout_seconds=30,source_kind='CONSTRUCTED',split_identity='NAME-CONSIST001-fresh-synthetic-actor',
            inputs=files,artifacts=['predictions.json'],
            metrics={'prediction_rows':'max','model_forwards':'min','optimizer_updates':'min'},budget={'model_forwards':count//8})
        write(d/'recipe.json',r)
        roster[d.name]=dict(arm=arm,seed=seed,endpoint=step,panel=split,adapter_sha256=weight_pin,actor_sha256=A,
            backbone_sha256=BASE,fit_case=None if checkp is None else fitcase,recipe_sha256=h(d/'recipe.json'),
            inputs={p:h(d/p) for p in files})
    for p in AS:stage('native',1371,0,format_source,None,'none',p)
    for cid,p in intake['runs'].items():
        run=fit_runs/p['id'];need(verify_run(run)['valid'] and h(run/'run.json')==p['receipt_sha256'],'fit seal changed')
        r=read(run/'run.json'); arm,x=cid.rsplit('-',1);seed=int(x)
        need(r['status']=='succeeded' and r['research_adequacy']=='EXECUTED_CRITERIA_MET' and r['seed']==seed,'native fit')
        for t in (128,1280):
            n=f'adapter-{t}.pt'
            for split in (('ordinary',) if t==128 else ('ordinary','contrast')):
                stage(arm,seed,t,cid,run/'output'/n,r['artifacts'][n],split)
    need(len(roster)==20,'wrong actor roster')
    write(out/'ledger.json',roster);print(h(out/'ledger.json'))
if __name__=='__main__':
    import sys
    need(len(sys.argv)==5,'three paths and exact fit seal digest required')
    fits,seal,runs=map(Path,sys.argv[1:4]);need(h(seal)==sys.argv[-1],'fit seal pin')
    main(fits,seal,runs)
