"""Deterministic TRAIN-only class counts; no model execution."""
import json,hashlib
from pathlib import Path

STUDY_SPEC={'study':'CSM-CONDITIONAL-018','arms':['local_learned','state_learned','local_exact'],'seeds':[101,202,303],'updates':1024,'segment_updates':64,'batch_size':16,'optimizer':'AdamW','learning_rate':.003,'clip_norm':1,'train_cases':512,'dev_cases':576,'gate':.95,'timing_case_selection':'first paired false/true cases of every DEV panel in sorted panel order','warm_repeats':5,'cold_repeats':3,'claim':'bounded fixed-budget comparison; no convergence or broad generalization claim'}

def event_counts(data):
 counts=[0]*7
 for cases in data['train'].values():
  for c in cases:
   counts[0]+=len(c['text'])
   for e in c['events']:
    if e['op']:counts[0]-=1;counts[e['op']]+=1
 if min(counts)<=0:raise ValueError('missing training operation')
 total=sum(counts)
 return {'counts':{str(i):n for i,n in enumerate(counts)},'weights':[total/(7*n) for n in counts]}
def timing_plan(data,data_sha256):
 return {'case_ids':[data['dev'][k][i]['case_id'] for k in sorted(data['dev']) for i in (0,1)],'data_sha256':data_sha256,'split':'dev','warm_repeats':5,'warmups':1,'cold_repeats':3,'cold_timeout_seconds':1800,'cold_order_seed':18018,'warm_order_seed':18017,'cold_batch':'entire selected roster in each process','order_seed':18017,'scope':'first paired false/true cases each18DEV panels, descriptive coverage not random population sample'}

def source_hashes(root):
 root=Path(root);files=list(root.glob('*.py'))+[root/'data.json',root/'EVENT_COUNTS.json',root/'PROTOCOL.md',root/'TIMING_PLAN.json',root/'RUNTIME_LOCK.json',root/'MATH.md']
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
if __name__=='__main__':
 p=Path(__file__).resolve().parent
 data=json.loads((p/'data.json').read_text())
 (p/'EVENT_COUNTS.json').write_text(json.dumps(event_counts(data),indent=2))
 (p/'TIMING_PLAN.json').write_text(json.dumps(timing_plan(data,hashlib.sha256((p/'data.json').read_bytes()).hexdigest()),indent=2))
