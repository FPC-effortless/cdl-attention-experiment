"""Stage-only scientific worker, native launch review mandatory."""
import json,os,random,sys,time,statistics,resource,tracemalloc
from pathlib import Path
import numpy as np
from learner import load,check_view
from execution_observer import write as observe_run
if sys.flags.optimize:raise RuntimeError('unsafe optimized Python')
OUT=Path(os.environ.get('TRAINLAB_OUTPUT_DIR','.'))
def save(n,x):(OUT/n).write_text(json.dumps(x,sort_keys=True,allow_nan=False))
def stage_fit():
 from learner import fit
 assert not any(Path(x).exists() for x in ('requests.json','truth.json'))
 plan=load('fit-settings.json');a=load('train.json');b=load('cal.json')
 assert len(a)==96 and len(b)==32 and check_view(a,load('train-forge-source.json'),('streams',)) and check_view(b,load('cal-forge-source.json'),('streams',))
 t=time.perf_counter();tc=time.process_time()
 model,m=fit(a,b,plan['sensor'],plan['threshold']);m.update(fit_wall_seconds=time.perf_counter()-t,fit_cpu_seconds=time.process_time()-tc)
 assert m['fit_rows']==96*3*126 and m['cal_rows']==32*3*126
 save('models.json',model);save('fit-summary.json',m);save('metrics.json',{'effective_rows':m['fit_rows']});observe_run({'rows':m['fit_rows'],'size':m['pred_trainable_coefficient_storage'],'stages':m['fitted_arms'],'wall_seconds':m['fit_wall_seconds']})
def stage_predict():
 begin=time.perf_counter()
 from serve import prepare,run,run_fused_group,ARMS
 assert not any(Path(x).exists() for x in ('train.json','cal.json','truth.json','settings.json','fit-settings.json'))
 data=load('requests.json')
 assert len(data)==192 and check_view(data,load('requests-forge-source.json'),('events','histories'))
 t=time.perf_counter_ns();model=prepare(load('models.json'));cold=time.perf_counter_ns()-t
 retained={a:{} for a in ARMS}
 for row in data:
  for arm in ARMS:retained[arm][row['id']]=run(row,model,arm)[0]
 timing=[];fall=0
 for size in (1,4,16):
  for rep in range(5):
   for packed in (bool(rep%2),not bool(rep%2)):
    for i in range(0,len(data),size):
     r=data[i:i+size];a=time.perf_counter_ns()
     result,fb=run_fused_group(r,model,packed);message=json.dumps(result,sort_keys=True,allow_nan=False);b=time.perf_counter_ns()
     assert result==[retained['predictive'][x['id']] for x in r]
     fall+=fb
     timing.append({'size':size,'packed':packed,'repeat':rep,'wave':i//size,'elapsed_ns':b-a,'requests':len(r),'json_bytes':len(message)})
 cold_trials={False:[],True:[]}; memory={}
 for j in range(20):
  for packed in (bool(j%2),not bool(j%2)):
   a=time.perf_counter_ns();m=prepare(load('models.json'));result,fb=run_fused_group(data[:16],m,packed)
   _=json.dumps(result,sort_keys=True,allow_nan=False);cold_trials[packed].append(time.perf_counter_ns()-a)
   assert result==[retained['predictive'][x['id']] for x in data[:16]]
 for packed in (False,True):
  tracemalloc.start();result,fb=run_fused_group(data[:16],model,packed)
  _=json.dumps(result,sort_keys=True,allow_nan=False);memory[packed]=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
 save('predictions.json',retained);save('timing.json',{'cold_model_prepare_ns':cold,'fresh_load_first_B16_ns':cold_trials,'incremental_tracemalloc_B16_bytes':memory,'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'fallbacks':fall,'records':timing})
 save('metrics.json',{'requests':len(data)});observe_run({'rows':len(data),'size':len(model['predictive'][-2]),'stages':len(ARMS),'wall_seconds':time.perf_counter()-begin})
def boot(values,seed=69068):
 cs=list(values);rng=random.Random(seed)
 out=sorted(sum(rng.choice(cs) for _ in cs)/len(cs) for __ in range(5000))
 return [out[124],out[4874]]
def stage_verify():
 begin=time.perf_counter()
 from serve import ARMS,observe
 d=load('requests.json');p=load('predictions.json');q=load('truth.json');timing=load('timing.json')
 assert {x['id'] for x in d}==set(q) and set(p)==set(ARMS) and all(set(v)==set(q) for v in p.values())
 by={a:{} for a in ARMS};target={False:set(),True:set()};participation={a:{False:[0,0],True:[0,0]} for a in ARMS};no_authority=0
 for x in d:
  row=x['id'];g=x['group'];fault=x['fault']; yy=q[row]['y']; assert g==q[row]['group'] and fault==q[row]['fault']
  eligible=observe(x)
  if not eligible:no_authority+=1
  if any(yy[k] for k in eligible):target[fault].add(g)
  for arm in ARMS:
   msg=p[arm][row];assert set(msg)=={'kind','lane'} and msg['kind']=='prediction_only_not_observed'
   lane=msg['lane'];assert lane in eligible or lane is None
   assert (lane is None)==(arm=='no-advice' or not eligible)
   hit=int(lane is not None and yy[lane]==1)
   by[arm].setdefault(g,[]).append(hit)
   participation[arm][fault][0]+=hit;participation[arm][fault][1]+=int(lane is not None)
 keys=sorted(by['predictive']);assert len(keys)==96 and all(len(v)==2 for a in ARMS for v in by[a].values())
 delta={a:boot([(sum(by['predictive'][g])-sum(by[a][g]))/2 for g in keys]) for a in ('slope','pca','raw')}
 cohorts_ok=all(len(target[f])>=30 for f in target)
 no_cost_mask=all(participation['predictive'][f][0]+.02*len(target[f])>=participation[a][f][0] for f in target for a in delta)
 # Timed stages emitted 15 matched repeat×wave designs and exact parity.
 rows=timing['records'];series={}
 def q95(x):return float(np.quantile(x,.95))
 def q50(x):return float(np.quantile(x,.5))
 def q99(x):return float(np.quantile(x,.99))
 for s in (1,4,16):
  keyed={}
  for arm in (False,True):
   views=[r for r in rows if r['size']==s and r['packed']==arm]
   assert len(views)==5*192//s and all(r['requests']==s for r in views)
   keyed[arm]={(r['wave'],r['repeat']):r['elapsed_ns']/1e6 for r in views}
  assert keyed[False].keys()==keyed[True].keys()
  series[s]={('packed' if k else 'folded_loop'):{'p50_wave_service_ms':q50(list(v.values())),'p95_wave_service_ms':q95(list(v.values())),'p99_wave_service_ms':q99(list(v.values())),'p95_per_request_amortized_ms':q95(list(v.values()))/s} for k,v in keyed.items()}
  # Resample whole wave identities, retaining five dependent repetitions.
  groups=sorted({w for w,_ in keyed[False]});rng=random.Random(69069);boost=[]
  for _ in range(5000):
   draw=[rng.choice(groups) for _ in groups];before=[keyed[False][(w,i)] for w in draw for i in range(5)];after=[keyed[True][(w,i)] for w in draw for i in range(5)]
   boost.append(1-q95(after)/q95(before))
  boost.sort();series[s]['p95_reduction_ci']=[boost[124],boost[4874]]
 cold0=q95(timing['fresh_load_first_B16_ns']['false']);cold1=q95(timing['fresh_load_first_B16_ns']['true'])
 memory_ok=timing['incremental_tracemalloc_B16_bytes']['true']<=8*1024**2
 pace=series[16]['p95_reduction_ci'][0]>=.10
 cold_ok=cold1<=1.05*cold0
 save('summary.json',{'scope':'CONSTRUCTED_CPU_NOT_QUALIFIED','groups':len(keys),'no_authority':no_authority,'event_support':{str(f):len(target[f]) for f in target},'per_arm_hits_and_active_by_fault':participation,'068_quality_gate':bool(cohorts_ok and no_cost_mask and all(v[0]>0 for v in delta.values())),'068_paired_CIs':delta,'069_exact_parity':True,'069_warm_target_B16_lower_ci_pass':series[16]['p95_reduction_ci'][0]>=.10,'069_cpu_mechanical_gate':bool(pace and cold_ok and memory_ok),'fresh_load_first_B16_p95_ns':{'folded_loop':cold0,'packed':cold1},'incremental_tracemalloc_B16_bytes':timing['incremental_tracemalloc_B16_bytes'],'process_peak_rss_kib_linux':timing['process_peak_rss_kib_linux'],'complete_software_wave_timings':series,'cold_load_and_prepare_common_ns':timing['cold_model_prepare_ns'],'fallbacks':timing['fallbacks']})
 save('metrics.json',{'groups':len(keys)});observe_run({'rows':len(keys),'size':len(series),'stages':len(delta),'wall_seconds':time.perf_counter()-begin})
if __name__=='__main__':
 if sys.argv[1:] not in (['fit'],['predict'],['verify']):raise RuntimeError('explicit stage only')
 {'fit':stage_fit,'predict':stage_predict,'verify':stage_verify}[sys.argv[1]]()
