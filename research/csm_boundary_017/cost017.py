"""CPU full-session timing. No training. Warm excludes imports/checkpoint load;
cold includes process startup/imports/load + selected full sessions, separately.
Uses full model.segment retaining per-token outputs; NOT an optimized streaming
API. State tensor bytes exclude outputs, Python metadata and allocator overhead.
Only caller-selected TRAIN/DEV case IDs accepted. No protected TEST interface.
"""
import argparse,json,time,statistics,subprocess,sys,hashlib,os,tempfile,math
from pathlib import Path

def session(model,text):
 import torch
 start=time.perf_counter_ns()
 raw=text.encode('ascii');x=torch.tensor([list(raw)],dtype=torch.long)
 initial=model.init(1);out=model.segment(x,initial)
 elapsed=(time.perf_counter_ns()-start)/1e9
 # Out remains live until timer ends; no grading/traces/raw audit replay in timer.
 return elapsed,out,len(raw)
def tensor_bytes(obj):
 import torch
 if isinstance(obj,torch.Tensor):return obj.nelement()*obj.element_size()
 if isinstance(obj,dict):return sum(tensor_bytes(v) for v in obj.values())
 if isinstance(obj,(list,tuple)):return sum(tensor_bytes(v) for v in obj)
 return 0
def warm(model,cases,repeats=5,warmups=1):
 import torch
 if not cases or repeats<2 or warmups<0:raise ValueError('nonempty workload and repeated timings required')
 if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('duplicate case ID')
 if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('CPU only')
 was=model.training;model.eval();rows=[]
 try:
  with torch.no_grad():
   for _ in range(warmups):
    for c in cases:session(model,c['text'])
   for repeat in range(repeats):
    for c in cases:
     seconds,out,n=session(model,c['text']);rows.append({'repeat':repeat,'case_id':c['case_id'],'seconds':seconds,'input_bytes':n,'persistent_state_tensor_bytes':tensor_bytes(out[2])})
 finally:model.train(was)
 return {'scope':'warm CPU full uninterrupted session: ASCII encoding, tensor construction, fresh state init, ingestion and all queries; excludes import/load/grading','case_median_seconds':{c['case_id']:statistics.median(r['seconds'] for r in rows if r['case_id']==c['case_id']) for c in cases},'arm':model.arm,'case_ids':[c['case_id'] for c in cases],'rows':rows,'median_seconds':statistics.median(r['seconds'] for r in rows),'total_measured_seconds':sum(r['seconds'] for r in rows),'threads':torch.get_num_threads(),'repeats':repeats,'warmups':warmups,'state_bytes_limit':'tensor payload only, not total RSS, allocator overhead or Python metadata'}
def paired_warm(left,right,required_ids):
 """Casewise cost ratios on a prospectively selected identical workload roster.
 Descriptive only; no pooling heterogeneous-length cases into a single claim.
 """
 if not required_ids or len(set(required_ids))!=len(required_ids):raise ValueError('unique nonempty timing roster')
 if left['case_ids']!=required_ids or right['case_ids']!=required_ids or left['repeats']!=right['repeats']:raise ValueError('paired timing roster mismatch')
 result=[]
 for cid in required_ids:
  a=[r for r in left['rows'] if r['case_id']==cid];b=[r for r in right['rows'] if r['case_id']==cid]
  if [r['repeat'] for r in a]!=list(range(left['repeats'])) or [r['repeat'] for r in b]!=list(range(right['repeats'])):raise ValueError('missing timing repeats')
  if len({r['input_bytes'] for r in a+b})!=1:raise ValueError('different workloads')
  x=statistics.median(r['seconds'] for r in a);y=statistics.median(r['seconds'] for r in b)
  if not math.isfinite(x) or not math.isfinite(y) or not x>0 or not y>0:raise ValueError('invalid timing')
  result.append({'case_id':cid,'input_bytes':a[0]['input_bytes'],'left_seconds':x,'right_seconds':y,'ratio_left_over_right':x/y})
 return {'scope':'paired same-case medians, not a heterogeneous pooled frontier','left_arm':left['arm'],'right_arm':right['arm'],'cases':result}
def selected(data,split,ids):
 if split not in ('train','dev') or not ids or len(set(ids))!=len(ids):raise ValueError('TRAIN/DEV case roster required')
 lookup={c['case_id']:c for rows in data[split].values() for c in rows}
 return [lookup[i] for i in ids]
def cold(checkpoint,arm,data_path,split,ids,repeats=3,timeout=120):
 if repeats<2:raise ValueError('repeated timings required')
 # Validate outside timed process; timing process reads source data again.
 selected(json.loads(Path(data_path).read_text()),split,ids)
 rows=[]
 with tempfile.TemporaryDirectory() as d:
  spec=Path(d)/'spec.json';spec.write_text(json.dumps({'checkpoint':str(Path(checkpoint).resolve()),'arm':arm,'data':str(Path(data_path).resolve()),'split':split,'ids':ids}))
  for repeat in range(repeats):
   begin=time.perf_counter_ns()
   try:
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',str(spec)],capture_output=True,text=True,timeout=timeout,check=True)
    payload=json.loads(p.stdout);row={'repeat':repeat,'seconds':(time.perf_counter_ns()-begin)/1e9,'status':'ok','child':payload}
   except (subprocess.SubprocessError,ValueError) as error:
    row={'repeat':repeat,'seconds':(time.perf_counter_ns()-begin)/1e9,'status':'failed','failure_type':type(error).__name__}
   rows.append(row)
 return {'scope':'cold-process wall time including interpreter startup, imports, data/checkpoint read, model construction/load, encoding and full selected sessions, result serialization and process exit; OS file cache NOT flushed','rows':rows,'timing_complete':all(r['status']=='ok' for r in rows),'median_seconds':statistics.median(r['seconds'] for r in rows) if all(r['status']=='ok' for r in rows) else None,'total_attempt_seconds':sum(r['seconds'] for r in rows),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),'case_ids':ids,'split':split}
def child(spec):
 import torch
 from model import Model,ARMS
 cfg=json.loads(Path(spec).read_text());torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 if cfg['arm'] not in ARMS:raise ValueError('unknown arm')
 model=Model(cfg['arm']);model.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True));model.eval()
 cases=selected(json.loads(Path(cfg['data']).read_text()),cfg['split'],cfg['ids']);rows=[]
 with torch.no_grad():
  for c in cases:
   elapsed,out,n=session(model,c['text']);rows.append({'case_id':c['case_id'],'session_seconds':elapsed,'input_bytes':n,'persistent_state_tensor_bytes':tensor_bytes(out[2])})
 return {'sessions':rows,'threads':1}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',required=True);args=p.parse_args();print(json.dumps(child(args.child)))
