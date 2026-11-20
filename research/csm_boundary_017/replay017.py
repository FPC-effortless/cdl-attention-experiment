"""Exact-bound endpoint replay; no training and no protected TEST."""
import json,hashlib
from pathlib import Path
import torch
from model import Model,ARMS
from evaluate import score_dataset
from audit017 import audit_result,FIELDS
from prepare017 import source_hashes

def replay(root,checkpoint,result_path,output,expected_checkpoint_sha256,expected_result_sha256):
 if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=expected_checkpoint_sha256 or hashlib.sha256(Path(result_path).read_bytes()).hexdigest()!=expected_result_sha256:raise ValueError('verified TrainLab artifact hash mismatch')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 root=Path(root).resolve()
 if root!=Path(__file__).resolve().parent:raise ValueError('replay source root must match imported module location')
 admission=json.loads((root/'ADMISSION_017.json').read_text())
 if admission['source_sha256']!=source_hashes(root):raise ValueError('replay source/data drift')
 result=json.loads(Path(result_path).read_text());arm=result['arm'];seed=result['seed']
 if arm not in ARMS or seed not in (101,202,303) or result['finished_update']!=1024:raise ValueError('incomplete endpoint')
 m=Model(arm);m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));m.eval();data=json.loads((root/'data.json').read_text())
 fresh={'arm':arm,'seed':seed,'dev':score_dataset(m,data['dev']),'train_probe':score_dataset(m,data['train'])}
 for split in ('dev','train_probe'):
  if fresh[split]!=result[split]:raise ValueError('endpoint replay differs: '+split)
 audit=audit_result(fresh,data,arm,seed)
 for split,panels in audit['panels'].items():
  for panel,a in panels.items():
   for field in FIELDS:
    observed=a['metrics'][field];retained=result[split][panel]['metrics'][field]
    if observed is None or retained is None:
     if observed!=retained:raise ValueError('independent undefined metric mismatch')
    elif abs(observed-retained)>1e-12:raise ValueError('independent scoring mismatch')
 report={'arm':arm,'seed':seed,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'original_result_sha256':hashlib.sha256(Path(result_path).read_bytes()).hexdigest(),'source_sha256':admission['source_sha256'],'exact_all_records_and_metrics_equal':True,'audit':audit,'scope':'same-host frozen endpoint execution + separately implemented saved-record scorer; not retraining or external reproduction'}
 Path(output).write_text(json.dumps(report));return report
