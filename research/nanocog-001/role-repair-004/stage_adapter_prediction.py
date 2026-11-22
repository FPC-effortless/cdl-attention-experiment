"""Prepare exact 13 no-update actor jobs after reviewed FIT only."""
import json,shutil
from hashlib import sha256
from pathlib import Path
from trainlab.runner import verify_run
R=Path(__file__).resolve().parent
A='6915a754aa598d5608f0041850b3897ecfeef0bd8731835815e24333ba6c98d1'
BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
ARMS=('literal_control','tap_compare');SEEDS=(1471,1571,1671)
def read(p):return json.loads(p.read_bytes())
def h(p):return sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')
def need(ok,why):
    if not ok:raise ValueError(why)
def main(fits,registry,fit_runs):
    intake=read(registry); parent=read(fits/'ledger.json')
    need(set(parent)==set(intake['runs'])=={f'{a}-{s}' for a in ARMS for s in SEEDS},'fit roster')
    out=R/'repair_actor_admissions';out.mkdir(exist_ok=False)
    roster={}; format_source=next(iter(parent))
    def stage(arm,seed,step,fitcase,checkp,weight_pin):
        src=fits/fitcase
        entry=parent[fitcase]
        d=out/f'{arm}-{seed}-{step}';d.mkdir()
        # No train.json or adapter supervision in any actor.
        files=[x for x in entry['inputs'] if x.startswith(('cog_comm/','nanocog/')) or x in
               ('tap_gate_residual.py','frozen_gate_adapter.py','fit_adapter.py','checkpoint.pt')]
        for f in files:
            need(h(src/f)==entry['inputs'][f],'predecessor source mismatch')
            p=d/f;p.parent.mkdir(exist_ok=True);shutil.copyfile(src/f,p)
        need(h(d/'checkpoint.pt')==BASE and h(R/'repair_panel/actor.json')==A,'actor or base changed')
        shutil.copyfile(R/'repair_panel/actor.json',d/'actor.json');shutil.copyfile(R/'predict_adapter.py',d/'predict_adapter.py')
        files+=['actor.json','predict_adapter.py']
        if checkp is not None:
            need(h(checkp)==weight_pin,'exact adapter digest')
            shutil.copyfile(checkp,d/'adapter.pt');files.append('adapter.pt')
        r=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
            name=d.name,description='Exact frozen/new adapter replay on prospective public synthetic actor.',
            command=['python','predict_adapter.py',arm,str(seed),str(step),weight_pin],
            seed=seed,timeout_seconds=30,source_kind='CONSTRUCTED',split_identity='ROLE-REPAIR004-fresh-synthetic-actor',
            inputs=files,artifacts=['predictions.json'],
            metrics={'prediction_rows':'max','model_forwards':'min','optimizer_updates':'min'},budget={'model_forwards':120})
        write(d/'recipe.json',r)
        roster[d.name]=dict(arm=arm,seed=seed,endpoint=step,adapter_sha256=weight_pin,actor_sha256=A,
            backbone_sha256=BASE,fit_case=None if checkp is None else fitcase,recipe_sha256=h(d/'recipe.json'),
            inputs={p:h(d/p) for p in files})
    stage('native',1371,0,format_source,None,'none')
    for cid,p in intake['runs'].items():
        run=fit_runs/p['id'];need(verify_run(run)['valid'] and h(run/'run.json')==p['receipt_sha256'],'fit seal changed')
        r=read(run/'run.json'); arm,x=cid.rsplit('-',1);seed=int(x)
        need(r['status']=='succeeded' and r['research_adequacy']=='EXECUTED_CRITERIA_MET' and r['seed']==seed,'native fit')
        for t in (128,1280):
            n=f'adapter-{t}.pt';stage(arm,seed,t,cid,run/'output'/n,r['artifacts'][n])
    need(len(roster)==13,'wrong actor roster')
    write(out/'ledger.json',roster);print(h(out/'ledger.json'))
if __name__=='__main__':
    import sys
    need(len(sys.argv)==5,'three paths and exact fit seal digest required')
    fits,seal,runs=map(Path,sys.argv[1:4]);need(h(seal)==sys.argv[-1],'fit seal pin')
    main(fits,seal,runs)
