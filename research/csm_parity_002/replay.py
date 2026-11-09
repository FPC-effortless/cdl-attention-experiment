"""Cold-process no-fit replay; load only this package's trusted checkpoints."""
import hashlib,json,zipfile
from pathlib import Path
import torch
from regulated import Regulated,base,ROOT
here=Path(__file__).resolve().parent
rows=json.loads((here/'pilot_results.json').read_text())['rows']
raw=zipfile.ZipFile(ROOT/'artifacts/csm-original-scope-reconstructed-20261109/CSM-GRAPH-001-verified-project.zip').read('CSM-GRAPH-001/csm-foundation/experiments/csm_foundation_001/data/input.txt')
assert hashlib.sha256(raw).hexdigest()=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
dev=torch.tensor(list(raw[1003854:1003854+1025])).unsqueeze(0)
torch.set_num_threads(1)
for row in rows:
 path=here/f"checkpoint-{row['seed']}-{row['variant']}.pt"
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['checkpoint_sha256']
 m=Regulated(base.Config(vocab_size=256),row['variant']).float();m.load_state_dict(torch.load(path,weights_only=False)['model'])
 with torch.no_grad():
  s=m.init_state(1,'cpu');loss=0
  for t in range(1024):
   y,s,_=m.step(dev[:,t],s);loss+=torch.nn.functional.cross_entropy(y,dev[:,t+1]).item()
 assert loss/1024==row['dev_nll']
print(json.dumps({'status':'PASS','cold_replays':len(rows),'training_updates':0}))
