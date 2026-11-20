"""Separate arithmetic/reference path for approved post-run audit only.

No learner execution. Does not import serving/worker/gateway. Do not run on
actual forecasts or truth before exact verifier-stage review and approval.
"""
from collections import defaultdict
from pathlib import Path
import json,math,random
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def unique_run(phase):
 runs=list((ROOT/'execution'/phase/'runs').iterdir());assert len(runs)==1
 return runs[0]/'output'
def legal(req):
 groups=defaultdict(list)
 for l,s,t,n,body in req['events']:
  if s==req['scope'] and t<=req['tick']:groups[l].append(((t,n),body))
 res=set()
 for k, events in groups.items():
  assert len({x[0] for x in events})==len(events)
  _,winner=max(events,key=lambda x:x[0])
  h=req['histories'].get(k,[])
  if winner==[1,True] and [v[0] for v in h]==list(range(req['tick']-2,req['tick']+1)):
   assert all(len(v[1])==16 and all(math.isfinite(x) for x in v[1]) for v in h)
   res.add(k)
 return res
def quant(s,p):
 s=sorted(s);x=p*(len(s)-1);i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
def boot(v):
 r=random.Random(69068);n=len(v)
 s=sorted(sum(r.choice(v) for _ in range(n))/n for i in range(5000))
 return [s[124],s[4874]]
def main():
 stage=unique_run('predict');ret=read(unique_run('verify')/'summary.json');case=read(ROOT/'frozen/requests.json');truth=read(ROOT/'frozen/truth.json');plan=read(stage/'predictions.json');clock=read(stage/'timing.json')
 hit=defaultdict(lambda:defaultdict(int));counts=defaultdict(lambda:{False:[0,0],True:[0,0]});eligible_case={False:0,True:0};empty=0
 for c in case:
  k=c['id'];lg=legal(c);f=c['fault'];g=c['group'];yes=truth[k]['y'];assert set(yes)==set(c['histories'])
  if any(yes[l] for l in lg):eligible_case[f]+=1
  empty+=not lg
  for arm,p in plan.items():
   advice=p[k]; lane=advice.get('lane');assert advice=={'kind':'prediction_only_not_observed','lane':lane}
   assert (lane is None)==(not lg or arm=='no-advice') and (lane is None or lane in lg)
   win=int(lane is not None and yes[lane]);hit[arm][g]+=win;counts[arm][f][0]+=win;counts[arm][f][1]+=lane is not None
 assert empty==ret['no_authority']
 assert {k:{str(b).lower():v for b,v in w.items()} for k,w in counts.items()}==ret['per_arm_hits_and_active_by_fault']
 assert {str(k):v for k,v in eligible_case.items()}==ret['event_support']
 guards=True
 for arm in ('slope','raw','pca'):
  values=[(hit['predictive'][g]-hit[arm][g])/2 for g in sorted(hit['predictive'])]
  interval=boot(values);assert interval==ret['068_paired_CIs'][arm]
  guards=guards and interval[0]>0 and all(counts['predictive'][f][0]+.02*eligible_case[f]>=counts[arm][f][0] for f in (False,True))
 assert ret['068_quality_gate']==bool(guards and min(eligible_case.values())>=30)
 expect={(s,p,r,w) for s in (1,4,16) for p in (False,True) for r in range(5) for w in range(192//s)}
 recs={(r['size'],r['packed'],r['repeat'],r['wave']):r for r in clock['records']}
 assert len(recs)==len(clock['records']) and set(recs)==expect and all(v['requests']==k[0] and v['elapsed_ns']>0 for k,v in recs.items())
 for s in (1,4,16):
  draws=range(192//s);rng=random.Random(69069);speed=[]
  for _ in range(5000):
   w=[rng.choice(draws) for _ in draws]
   times=[[recs[s,a,r,k]['elapsed_ns']/1e6 for k in w for r in range(5)] for a in (False,True)]
   speed.append(1-quant(times[1],.95)/quant(times[0],.95))
  speed.sort();v=ret['complete_software_wave_timings'][str(s)]['p95_reduction_ci']
  assert max(abs(x-y) for x,y in zip([speed[124],speed[4874]],v))<1e-10
 c=clock['fresh_load_first_B16_ns'];assert all(len(c[k])==20 for k in ('true','false'))
 assert ret['069_cpu_mechanical_gate']==(ret['complete_software_wave_timings']['16']['p95_reduction_ci'][0]>=.10 and quant(c['true'],.95)<=1.05*quant(c['false'],.95) and clock['incremental_tracemalloc_B16_bytes']['true']<=8*1024**2)
 print('PASS independent authority/shape/paired 068 outcomes, all timed-wave IDs and 069 interval/gates; no learner rerun')
if __name__=='__main__':main()
