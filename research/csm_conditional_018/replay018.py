"""Exact-bound reload/replay; all panels retained, not independent retraining."""
import hashlib,json
from pathlib import Path
import torch
from model import Model
from evaluate import score_dataset
from prepare018 import source_hashes
def replay(root,checkpoint,result_path,output,checkpoint_sha256,result_sha256):
 root=Path(root).resolve()
 if root!=Path(__file__).resolve().parent:raise ValueError('import root mismatch')
 for p,h in [(checkpoint,checkpoint_sha256),(result_path,result_sha256)]:
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('artifact drift')
 from runtime018 import verify
 verify(root)
 admission=json.loads((root/'ADMISSION_018.json').read_text())
 if admission['source_sha256']!=source_hashes(root):raise ValueError('source/data drift')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 old=json.loads(Path(result_path).read_text());m=Model(old['arm']);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.eval();data=json.loads((root/'data.json').read_text())
 fresh={k:score_dataset(m,data[v]) for k,v in [('dev','dev'),('train_probe','train')]}
 if any(fresh[k]!=old[k] for k in fresh):raise ValueError('replay mismatch')
 from audit018 import audit
 audited=audit(data,dict(old,**fresh))
 report=dict(audit=audited,checkpoint_sha256=checkpoint_sha256,result_sha256=result_sha256,source_sha256=admission['source_sha256'],exact_replay=True,scope='same-host checkpoint reload, not retraining; independent saved-record scorer included')
 Path(output).write_text(json.dumps(report));return report
