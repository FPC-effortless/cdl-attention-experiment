"""Seal replay logits only after native and captured feature parity."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from trainlab.runner import verify_run
PIN='2a364701848636d70e40719b9f85ab1a8bba7f93666954f9280d69f212b06828'
S='f868706de68fd723e0e87388969468763cd9274d5fae4739d173cb6107f262df'
def read(p):return json.loads(p.read_bytes())
def h(p):return sha256(p.read_bytes()).hexdigest()
def need(x,m):
    if not x:raise ValueError(m)
def main():
    a=Path('readout_admissions')
    need(h(a/'ledger.json')==PIN and h(Path('TRACE_SEAL.json'))==S,'exact prior seal')
    prior=read(Path('TRACE_SEAL.json'))['runs'];stage=read(a/'ledger.json');out={}
    need(set(stage)=={'readout-1171','readout-1371'},'roster')
    for k,c in stage.items():
      src=a/k
      r=read(src/'cli-output.json');run=Path('readout_runs')/r['id']
      need(verify_run(run)['valid'] and r==read(run/'run.json') and r['name']==k
          and r['inputs']==c['inputs'] and r['seed']==c['seed'] and r['status']=='succeeded'
          and r['metrics']==dict(prediction_rows=1536,model_forwards=192,optimizer_updates=0),'native receipt')
      need(h(src/'recipe.json')==c['recipe_sha256'] and all(h(src/f)==q for f,q in c['inputs'].items()),'native closure')
      p=run/'output/predictions.json';q=read(p)
      need(q['actor_sha256']==c['actor_sha256'] and q['checkpoint_sha256']==c['checkpoint_sha256']
          and q['seed']==c['seed'] and h(p)==r['artifacts']['predictions.json'],'outputs')
      prev=prior[f"trace-{c['seed']}"];old=Path('trace_runs')/prev['id']/'output/TRACE.f32le'
      need(h(old)==prev['artifacts']['TRACE.f32le'],'sealed TRACE changed')
      vh=np.frombuffer(old.read_bytes(),dtype='<f4').reshape(1536,132)[:,-32:].copy(order='C')
      need(sha256(vh.tobytes()).hexdigest()==q['pre_output_sha256'],'native pre-output trace not byte-exact')
      out[k]=dict(id=r['id'],receipt_sha256=h(run/'run.json'),prediction_sha256=h(p),
                  pre_output_sha256=q['pre_output_sha256'])
    file=Path('READOUT_SEAL.json')
    with file.open('x') as f:json.dump(dict(ledger_sha256=PIN,runs=out,original_trace_seal_sha256=S),f,
         sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(h(file))
if __name__=='__main__':main()
