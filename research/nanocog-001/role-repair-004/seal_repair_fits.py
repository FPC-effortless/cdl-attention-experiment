"""Read-only after all six native fits; verify complete exact roster."""
import json
from pathlib import Path
from hashlib import sha256
from trainlab.runner import verify_run
PIN='1bfd2de5842ced47984603cf8d6be24277808ca37bae1b1e6150b42475c86b2c'
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def need(ok,m):
    if not ok:raise ValueError(m)
p=Path('fit_admissions_v2');q=Path('fit_runs')
need(h(p/'ledger.json')==PIN,'fit ledger changed')
u=read(p/'ledger.json');out={}
summ={}
need(len(u)==6,'fit roster')
for name,a in u.items():
    src=p/name;r=read(src/'cli-output.json');run=q/r['id']
    need(verify_run(run)['valid'] and r==read(run/'run.json') and r['status']=='succeeded' and
         r['research_adequacy']=='EXECUTED_CRITERIA_MET' and r['inputs']==a['inputs'] and r['seed']==a['seed'] and
         0<r['elapsed_seconds']<=180,'bad native fit')
    need(read(src/'recipe.json')==read(run/'experiment.json') and h(src/'recipe.json')==a['recipe_sha256'] and
         all(h(src/k)==v for k,v in a['inputs'].items()),'admission drift')
    f=read(run/'output/fit.json')
    need(f['plan']['arm']==a['arm'] and f['plan']['seed']==a['seed'] and
         f['backbone_sha256']==a['frozen_base_sha256'] and f['train_sha256']==a['train_sha256'] and
         set(f['checkpoints'])=={'128','1280'},'fit lineage')
    a0=f['adapter_initial_sha256']; sched=f['schedule_sha256']
    need(summ.setdefault(r['seed'],(a0,sched))==(a0,sched),'matched initialization/schedule lost')
    out[name]=dict(id=r['id'],receipt_sha256=h(run/'run.json'),artifacts=r['artifacts'])
need(len({a['id'] for a in out.values()})==6,'duplicate')
d=Path('FIT_SEAL.json')
with d.open('x') as f:json.dump(dict(ledger_sha256=PIN,runs=out),f,sort_keys=True,indent=2,allow_nan=False)
print(h(d))
