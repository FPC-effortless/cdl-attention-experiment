"""Warm complete-session timing, interleaved arms. No benchmark before qualification."""
import json,time,random,hashlib
from pathlib import Path
import torch
from model import Model,tensor,ARMS
def verify_inputs(root,endpoints):
 from prepare019 import source_hashes
 from runtime019 import verify
 root=Path(root);verify(root);a=json.loads((root/'TIMING_ADMISSION.json').read_text())
 if a.get('approved') is not True or not a.get('reviewer') or a.get('author')==a['reviewer']:raise ValueError('independent timing review required')
 if a['source_sha256']!=source_hashes(root):raise ValueError('timing source drift')
 if a['seed'] not in (101,202,303) or set(endpoints)!=set(ARMS) or set(a['endpoints'])!=set(ARMS):raise ValueError('timing arm/seed roster')
 for arm,p in endpoints.items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=a['endpoints'][arm]['sha256']:raise ValueError('checkpoint mismatch')
 return a
def check_output(a,arm,caseid,out):
 observed={'answers':out[1].tolist(),'final_state':[q.tolist() for q in out[2][0]]}
 if observed!=a['endpoints'][arm]['expected'][caseid]:raise ValueError('timing output differs from independently reviewed checkpoint replay')
 return observed
def warm(root,endpoints,output):
 root=Path(root);admission=verify_inputs(root,endpoints);plan=json.loads((root/'TIMING_PLAN.json').read_text());data=json.loads((root/'data.json').read_text())
 if hashlib.sha256((root/'data.json').read_bytes()).hexdigest()!=plan['data_sha256']:raise ValueError('data drift')
 roster={c['case_id']:c for rows in data['dev'].values() for c in rows};models={}
 if set(endpoints)!=set(ARMS):raise ValueError('all arms required')
 for arm,p in endpoints.items():
  m=Model(arm);m.load_state_dict(torch.load(p,weights_only=True));m.eval();models[arm]=m
 rng=random.Random(18017);records=[];torch.set_num_threads(1)
 with torch.inference_mode():
  for repeat in range(6):
   for caseid in plan['case_ids']:
    order=list(ARMS);rng.shuffle(order)
    for arm in order:
     begin=time.perf_counter_ns();record=dict(case_id=caseid,arm=arm,repeat=repeat,order=order,warmup=repeat==0)
     try:
      m=models[arm];out=m.segment(tensor([roster[caseid]]),m.init(1));record['elapsed_ns']=time.perf_counter_ns()-begin
      observed=check_output(admission,arm,caseid,out);record.update(status='succeeded',output=observed)
     except Exception as error:
      record.update(status='failed',elapsed_ns=time.perf_counter_ns()-begin,error=repr(error));records.append(record);Path(output).write_text(json.dumps({'complete':False,'records':records}));raise
     records.append(record);Path(output).write_text(json.dumps({'complete':False,'records':records}))
 Path(output).write_text(json.dumps({'complete':True,'records':records}));return records

def cold(root,endpoints,output):
 """Three per-arm processes, randomized interleaved order, complete roster each."""
 import subprocess,sys,tempfile
 root=Path(root).resolve();admission=verify_inputs(root,endpoints);plan=json.loads((root/'TIMING_PLAN.json').read_text());rng=random.Random(18019);records=[]
 if set(endpoints)!=set(ARMS):raise ValueError('all arms required')
 for repeat in range(3):
  order=list(ARMS);rng.shuffle(order)
  for arm in order:
   with tempfile.TemporaryDirectory() as td:
    result=Path(td)/'result.json'
    cmd=[sys.executable,str(Path(__file__).resolve()),'cold_worker',str(root),arm,str(Path(endpoints[arm]).resolve()),str(result)]
    start=time.perf_counter_ns();record=dict(arm=arm,repeat=repeat,order=order,checkpoint_sha256=hashlib.sha256(Path(endpoints[arm]).read_bytes()).hexdigest())
    try:
     completed=subprocess.run(cmd,capture_output=True,text=True,timeout=1800);record.update(elapsed_ns=time.perf_counter_ns()-start,returncode=completed.returncode,stderr=completed.stderr)
     if completed.returncode:raise RuntimeError('cold process exited nonzero')
     record.update(status='succeeded',result=json.loads(result.read_text()))
    except Exception as error:
     record.update(status='failed',elapsed_ns=time.perf_counter_ns()-start,error=repr(error));records.append(record);Path(output).write_text(json.dumps({'complete':False,'records':records}));raise
    records.append(record);Path(output).write_text(json.dumps({'complete':False,'records':records}))
 Path(output).write_text(json.dumps({'complete':True,'records':records}));return records

def cold_worker(root,arm,checkpoint,output):
 import resource
 from runtime019 import verify
 root=Path(root);verify(root);admission=json.loads((root/'TIMING_ADMISSION.json').read_text())
 from prepare019 import source_hashes
 if admission['source_sha256']!=source_hashes(root) or hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=admission['endpoints'][arm]['sha256']:raise ValueError('cold source/checkpoint drift')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 plan=json.loads((root/'TIMING_PLAN.json').read_text());data=json.loads((root/'data.json').read_text())
 if hashlib.sha256((root/'data.json').read_bytes()).hexdigest()!=plan['data_sha256']:raise ValueError('data drift')
 roster={c['case_id']:c for rows in data['dev'].values() for c in rows};m=Model(arm);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.eval();records=[]
 with torch.inference_mode():
  for caseid in plan['case_ids']:
   out=m.segment(tensor([roster[caseid]]),m.init(1));check_output(admission,arm,caseid,out);records.append(dict(case_id=caseid,answers=out[1].tolist(),final_state=[q.tolist() for q in out[2][0]]))
 Path(output).write_text(json.dumps(dict(records=records,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Linux process highwater RSS including imports/checkpoint; not incremental tensor memory')))
if __name__=='__main__':
 import sys
 if len(sys.argv)==6 and sys.argv[1]=='cold_worker':cold_worker(*sys.argv[2:])
 else:raise SystemExit('cold_worker root arm checkpoint output')
