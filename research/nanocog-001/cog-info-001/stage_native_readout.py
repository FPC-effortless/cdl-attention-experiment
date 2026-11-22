"""Exact native readout recipes from the existing trace closure; no model calls."""
from hashlib import sha256
from pathlib import Path
import json,shutil
from trainlab.runner import verify_run
D=Path(__file__).resolve().parent
L='b78161b7687aa0d89c86e5192dd2863ca7542b5e905d2ccb4836143421fd4425'
S='f868706de68fd723e0e87388969468763cd9274d5fae4739d173cb6107f262df'
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
need=lambda b,m:None if b else (_ for _ in ()).throw(ValueError(m))
def main():
 src=D/'trace_admissions';z=D/'TRACE_SEAL.json'
 need(h(src/'ledger.json')==L and h(z)==S,'exact trace inputs')
 box=D/'readout_admissions';box.mkdir(exist_ok=False)
 sealed=read(z)['runs'];out={}
 for label,c in read(src/'ledger.json').items():
    r=D/'trace_runs'/sealed[label]['id']
    need(verify_run(r)['valid'] and h(r/'run.json')==sealed[label]['receipt_sha256'],'prior trace receipt')
    root=src/label
    name=f"readout-{c['seed']}"
    d=box/name;d.mkdir()
    keep=[x for x in c['inputs'] if x not in ('INFO001_CAPTURE_PROTOCOL.md','capture_stage_trace.py')]
    for x in keep:
       need(h(root/x)==c['inputs'][x],'trace source changed')
       p=d/x;p.parent.mkdir(exist_ok=True);shutil.copyfile(root/x,p)
    for x in ('READOUT_CHECK_LOCK.md','run_native_readout.py'):
       shutil.copyfile(D/x,d/x);keep.append(x)
    spec=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
      name=name,description='Native decoder reference without any adaptation or labels.',
      command=['python','run_native_readout.py',c['checkpoint_sha256'],str(c['seed'])],
      seed=c['seed'],timeout_seconds=45,source_kind='CONSTRUCTED',split_identity='INFO001-identical-readout-replay',
      inputs=keep,artifacts=['predictions.json'],metrics={'prediction_rows':'max','model_forwards':'min','optimizer_updates':'min'},
      budget={'model_forwards':192})
    save(d/'recipe.json',spec)
    out[name]=dict(seed=c['seed'],actor_sha256=c['actor_sha256'],checkpoint_sha256=c['checkpoint_sha256'],
       recipe_sha256=h(d/'recipe.json'),inputs={x:h(d/x) for x in keep})
 save(box/'ledger.json',out);print(h(box/'ledger.json'))
if __name__=='__main__':main()
