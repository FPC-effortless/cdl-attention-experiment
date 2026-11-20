"""Prospective resource arithmetic from software calibration, never outcome tuning."""
import argparse,json,zipfile,hashlib,importlib.util,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--data',required=True);p.add_argument('--generator',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 z=zipfile.ZipFile(a.archive);name=next(n for n in z.namelist() if n.endswith('/adapter/calibration.json'));c=json.loads(z.read(name))
 spec=importlib.util.spec_from_file_location('budget_generator',a.generator);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 def work(rows):
  return sum(max(r['answers'][q]['start']-(0 if q==0 else r['answers'][q-1]['end']+1) for r in rows) for q in range(6))
 measurements={(r['condition'],r['kind'],r['steps'],r['delay']):r for r in c['measurements'] if r['batch_size']==32}
 fixturework={delay:work([r for i in range(16) for r in g.episode_pair('software',900000+i,delay)]) for delay in [256,2048]}
 axes={};data=Path(a.data)
 for path in sorted(data.glob('dev_*.jsonl')):
  rows=[json.loads(line) for line in path.read_text().splitlines()]
  if len(rows)!=256:raise ValueError('full axis roster256 required')
  pairs={}
  for r in rows:pairs.setdefault(r['pair_id'],[]).append(r)
  ordered=[r for pair in pairs.values() for r in sorted(pair,key=lambda r:r['variant'])]
  batchinfo=[]
  for start in range(0,256,32):
   group=ordered[start:start+32];delay=2048 if max(r['requested_delay_bytes'] for r in group)>256 else 256
   scale=max(1.,work(group)/fixturework[delay]);cost={}
   for condition in ['native','exact']:
    for kind,steps in [('ar',None),('diffusion',4),('diffusion',8),('diffusion',16)]:
     m=measurements[(condition,kind,steps,delay)]
     # Resolved supplied-state is a new adapter: reserve twice measured raw parse/snapshot cost.
     conditioning=m['conditioning_seconds']*scale*(2 if condition=='exact' else 1)
     cost[f'{condition}:{kind}:{steps}']=m['seconds']-m['conditioning_seconds']+conditioning
   batchinfo.append({'maximum_requested_delay':delay,'external_byte_step_scale':scale,'cost_seconds':cost})
  distribution={str(d):sum(r['requested_delay_bytes']==d for r in rows) for d in sorted(set(r['requested_delay_bytes'] for r in rows))}
  axes[path.stem]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sessions':256,'pairs':128,'delay_sessions':distribution,'batches':batchinfo}
 primary_keys=['native:ar:None','native:diffusion:8','exact:ar:None','exact:diffusion:8']
 primary=3*sum(sum(b['cost_seconds'][k] for k in primary_keys) for axis in axes.values() for b in axis['batches'])
 # Fixed diagnostic proposal: first16pairs/axis, everyseed; K4/K16 full; AR/K8 zero+swapped.
 diagnostic=3*sum(sum(axis['batches'][0]['cost_seconds'][k] for k in ['native:diffusion:4','native:diffusion:16','exact:diffusion:4','exact:diffusion:16'])+2*sum(axis['batches'][0]['cost_seconds'][k] for k in primary_keys) for axis in axes.values())
 train={};cold={};warm_max={}
 for kind in ['ar','diffusion']:
  n=next(n for n in z.namelist() if n.endswith('/'+kind+'_timing/updates.jsonl'));r=[json.loads(x) for x in z.read(n).splitlines()]
  cold[kind]=r[0]['step_elapsed_before_trace_write_seconds'];warm_max[kind]=max(x['step_elapsed_before_trace_write_seconds'] for x in r[1:]);train[kind]=.02
 cache=3*(8192/32)*c['cache_calibration']['seconds']+3*(8192/32)*.25
 diag_by_axis={name:3*(sum(axis['batches'][0]['cost_seconds'][k] for k in ['native:diffusion:4','native:diffusion:16','exact:diffusion:4','exact:diffusion:16'])+2*sum(axis['batches'][0]['cost_seconds'][k] for k in primary_keys)) for name,axis in axes.items()}
 selected_diagnostic=diag_by_axis['dev_id']+diag_by_axis['dev_joint']
 proposals=[]
 for updates in [4096,3072]:
  training=12*updates*.02+12*2.102
  base=cache+training+2*primary+selected_diagnostic+300
  proposals.append({'updates_per_fit':updates,'training_seconds':training,'cache_seconds':cache,'primary_seconds':primary,'full_primary_replay_seconds':primary,'diagnostic_seconds':selected_diagnostic,'load_serialization_source_reserve_seconds':300,'planned_seconds':base,'remaining_inside_plan_cap_seconds':15037.6665-base,'fits_15037_6665_plan_cap':base<=15037.6665})
 out={'scope':'prospective estimates, not guarantees; raw software calibration, no scientific outcomes','calibration_archive_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),'remaining_seconds':20050.222,'planning_cap_seconds':15037.6665,'axes':axes,'warm_seconds_per_update_allowance':train,'observed_warm_step_max':warm_max,'observed_first_step_seconds':cold,'diagnostic_axis_seconds':diag_by_axis,'selected_diagnostic_axes':['dev_id','dev_joint'],'diagnostic_pair_selection':'first16pairs (32sessions) eachselectedaxis, all3seeds; AR/K8 zero+swapped, K4/K16full; selected before outcomes','proposals':proposals,'assumptions':['Current order means mixed1024/2048 batches cost2048; no fabricated per-session latency','Each primary evaluation repeated once independently across every seed and arm on ALL6axes','Only dev_id and dev_joint diagnostics, first16pairs each; diagnostics raw rescored but not independently rerun in this estimate','New resolved control reserves2x raw conditioning time; cache adds0.25s per32sessions; must retain measured actualcost and stop budget if excess','One20update software trace perheadfamily: measured warm maxima below0.02s; use0.02 perupdate and2.102 cold perfit; not statistical runtime guarantee','300seconds load/serialization/source operating reserve; separate25percent quota recovery margin remains','Previous all6axis diagnostic proposal exceededplanningcap; twoaxis allocation chosen prospectively, not by outcomes']}
 # Reviewer alternative, not an adopted freeze: deterministic first2048 complete TRAIN pairs.
 trainrows=[json.loads(x) for x in (data/'train.jsonl').read_text().splitlines()]
 trainpairs={}
 for r in trainrows:trainpairs.setdefault(r['pair_id'],[]).append(r)
 selected=[r for pair in list(trainpairs.values())[:2048] for r in sorted(pair,key=lambda r:r['variant'])]
 if len(selected)!=4096 or any(len(p)!=2 for p in list(trainpairs.values())[:2048]):raise ValueError('selected complete pair count')
 from collections import Counter
 selected_delay=Counter(r['requested_delay_bytes'] for r in selected)
 query_kinds=Counter(q['kind'] for r in selected for q in r['answers'])
 query_identity=Counter(q['identity'] for r in selected for q in r['answers'])
 operations=Counter(o['op'] for r in selected for o in r['operations'])
 per_role={op:dict(Counter(o['dst'] for r in selected for o in r['operations'] if o['op']==op)) for op in operations}
 cache_half=cache/2;train_est=12*4096*.02+12*2.102
 subtotal=cache_half+train_est+2*primary+diagnostic
 out['reviewer_alternative']={'status':'prospective alternative awaiting review, not adopted by this script','selection':'first2048complete TRAIN pairs in existing file order, bothvariants; no outcomes examined','train_sessions':len(selected),'train_pairs':2048,'train_queries':sum(len(r['answers']) for r in selected),'selected_ids_sha256':hashlib.sha256("\n".join(r['id'] for r in selected).encode()).hexdigest(),'requested_delay_sessions':dict(selected_delay),'query_kind_counts':dict(query_kinds),'query_identity_counts':dict(query_identity),'operation_counts':dict(operations),'destination_identity_counts_by_operation':per_role,'max_external_batch_byte_steps':max(work(selected[i:i+32]) for i in range(0,len(selected),32)),'cache_seconds':cache_half,'training_seconds':train_est,'updates_per_fit':4096,'answer_exposures_per_fit':524288,'primary_seconds':primary,'full_primary_replay_seconds':primary,'diagnostic_seconds_all6axes':diagnostic,'subtotal_seconds':subtotal,'execution_overhead10percent_seconds':subtotal*.1,'planned_seconds':subtotal*1.1,'planning_cap_remaining_seconds':15037.6665-subtotal*1.1,'fits_plan':subtotal*1.1<=15037.6665,'scope':'All6axes primary+replay and all6axes first16pairs diagnostic roster retained; reduced distinctTRAIN data means greater repetition, not same data exposure as fullTRAIN proposal;10percent overhead replaces fixed300second reserve,25percent quota buffer remains'}
 Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='axes'},indent=2))
if __name__=='__main__':main()
