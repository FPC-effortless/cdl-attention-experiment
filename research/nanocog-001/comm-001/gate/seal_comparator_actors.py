"""Verify and seal exact comparator actor receipts without judging predictions."""
import json
from pathlib import Path
from hashlib import sha256
import sys
from trainlab.runner import verify_run
root=Path('actor-admissions-comparators'); runs=Path('actors-comparators')
def read(p):return json.loads(p.read_bytes())
def h(p):return sha256(p.read_bytes()).hexdigest()
ledger_path=root/'ledger.json'
if h(ledger_path)!=sys.argv[1]:
    raise ValueError('wrong independently reviewed actor ledger')
ledger=read(ledger_path); out={}
if len(ledger)!=36:
    raise ValueError('actor coverage')
for cid,tr in ledger.items():
    case=root/cid
    r=read(case/'cli-output.json')
    run=runs/r['id']
    assert tr['recipe_sha256']==h(case/'recipe.json') and read(case/'recipe.json')==read(run/'experiment.json')
    assert verify_run(run)['valid'] and r==read(run/'run.json') and r['status']=='succeeded'
    assert r['name']==cid and r['seed']==tr['seed'] and r['inputs']==tr['inputs']
    assert r['metrics']['prediction_rows']==(960 if tr['split']=='control' else 2560)
    out[cid]={'id':r['id'],'run_sha256':h(run/'run.json'),'prediction_sha256':r['artifacts']['predictions.json'],'elapsed_seconds':r['elapsed_seconds']}
assert len(out)==36 and len({v['id'] for v in out.values()})==36
with open('ACTOR_COMPARATORS_SEAL.json','x') as f:json.dump(out,f,sort_keys=True,indent=2)
print(h(Path('ACTOR_COMPARATORS_SEAL.json')))
