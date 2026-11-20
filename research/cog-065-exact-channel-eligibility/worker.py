"""Deterministic sender and saved-token verifier; no learner or third-party effects."""
import itertools,json,os,statistics,sys,time
from collections import defaultdict
from pathlib import Path
from codec import CLAUSES,ORDER,EOS,encode,parse,message_length
from rules import ROLES,KNOWN,receive,reference
from timeline import Envelope,snapshot
from generate import decode
from primitive_reference import World,initial,step,done
if sys.flags.optimize:raise RuntimeError('integrity guards require nonoptimized Python')
OUT=Path(os.environ.get('TRAINLAB_OUTPUT_DIR','.'))
def read(p):return json.loads(Path(p).read_text())
def write(p,v): (OUT/p).write_text(json.dumps(v,sort_keys=True,allow_nan=False))

def key(role,facts):return (role,tuple(facts[k] for k in ORDER))
def choose(role,facts,budgets=(16,64)):
 # For each virtual context in the public roster, reward matching complete
 # expected action from the visible facts. No actual private-context records.
 locals=[{k:facts[k] for k in cols} for cols in KNOWN[role]]
 expected=receive(role,{},facts)
 best={x:None for x in budgets}
 for n in range(len(ORDER)+1):
  for cols in itertools.combinations(ORDER,n):
   msg=encode(facts,cols,64);cost=message_length(msg)
   bits={k:facts[k] for k in cols}
   wins=sum(receive(role,c,bits)==expected for c in locals)
   rank=(-wins,cost,msg)
   for b in budgets:
    if cost<=b and (best[b] is None or rank<best[b]):best[b]=rank
 return {str(b):best[b][2] for b in budgets}
def compile_cache():
 cache={}
 for vals in itertools.product(*(CLAUSES[k] for k in ORDER)):
  f=dict(zip(ORDER,vals))
  for r in ROLES:cache[key(r,f)]=choose(r,f)
 return cache
def occupancy(f):
 return snapshot([Envelope(**r) for r in f['raw']],f['cutoff'],decode)
def emit():
 assert not Path('private-reference.json').exists()
 rows=read('observations.json')['rows'];assert len(rows)==144
 start=time.perf_counter_ns();cache=compile_cache();build=time.perf_counter_ns()-start
 # Serialized bytes are a reproducible lower-bound storage estimate, not RSS.
 cache_json=json.dumps([(k,v) for k,v in cache.items()],separators=(',',':'))
 messages={}; warm=[]
 for repeat in range(5):
  for i,row in enumerate(rows):
   a=time.perf_counter_ns();f=row['features'];facts=occupancy(f);b=time.perf_counter_ns()
   candidates=cache[key(f['role'],facts)];m=candidates['16'];c=time.perf_counter_ns()
   assert parse(m,16)=={k:facts[k] for k in parse(m,16)}
   warm.append({'row':row['id'],'repeat':repeat,'acq_ns':b-a,'lookup_ns':c-b})
   msg={'empty':EOS,'bound16':m,'full':candidates['64']}
   if repeat==0:messages[row['id']]=msg
   else:assert msg==messages[row['id']]
 # Cold uncached 16/full solve on first 12 source IDs, no arbitrary threshold use.
 cold=[]
 for r in rows[:12]:
  f=r['features'];v=occupancy(f)
  a=time.perf_counter_ns();full=choose(f['role'],v);cold.append(time.perf_counter_ns()-a)
  assert full==cache[key(f['role'],v)]
 assert len(messages)==144
 write('messages.json',messages);(OUT/'cache.json').write_text(cache_json);write('clock.json',{'cache_entries':len(cache),'cache_bytes_json':len(cache_json.encode()),'compile_ns':build,'warm':warm,'cold_choose_ns':cold})
 write('metrics.json',{'messages':len(messages)})
def valid_transition(act,q):
 p=q['primitive'];w=World(((0,p['target']),),(2,),('parcel request',));s=initial(w)
 for a in p['path']:
  result=step(w,s,a)
  if result is None:raise ValueError('bad primitive witness')
  s,_=result
 assert p['complete']==done(w,s)
 if act in ('close','attest','notify complete'):return done(w,s)
 if act=='resume':
  cand=[4,p['target']][s[-1]==0 and s[0]>0]
  return step(w,s,cand) is not None and not done(w,s) and q['facts']['authority']=='current'
 return True
def interval(values):
 import random
 v=tuple(values);rng=random.Random(65065)
 samples=sorted(sum(rng.choice(v) for _ in v)/len(v) for _ in range(5000))
 return [samples[124],samples[4874]]
def verify():
 obs=read('observations.json')['rows'];q=read('private-reference.json');m=read('messages.json');clock=read('clock.json')
 ids={r['id'] for r in obs};assert len(ids)==144 and ids==set(q)==set(m)
 cache={(k[0],tuple(k[1])):v for k,v in read('cache.json')};assert len(cache)==clock['cache_entries']
 tests={};units=defaultdict(list);expected_keys={'empty','bound16','full'}
 for row in obs:
  i=row['id'];z=q[i];f=row['features'];m0=m[i];assert set(m0)==expected_keys and m0['empty']==EOS
  facts=occupancy(f);assert facts==z['facts'] and z['role']==f['role'] and z['group']==row['groups']['causal_group']
  assert len(z['contexts'])==4 and z['contexts']==[{k:facts[k] for k in cols} for cols in KNOWN[z['role']]]
  gold=reference(z['role'],facts);assert valid_transition(gold,z)
  out={a:{'correct':0,'danger':0,'tokens':message_length(m0[a])} for a in expected_keys}
  for a,budget in [('empty',1),('bound16',16),('full',64)]:
   t0=time.perf_counter_ns();decoded=parse(m0[a],budget)
   assert all(facts[k]==v for k,v in decoded.items())
   for private in z['contexts']:
    act=receive(z['role'],private,decoded);ok=valid_transition(act,z)
    out[a]['correct']+=int(act==gold and ok)
    out[a]['danger']+=int(act in ('close','attest','notify complete','resume') and (act!=gold or not ok))


  units[z['group']].append(out);tests[i]={'role':z['role'],'gold':gold,'scores':out}
 assert len(units)==72 and all(len(rows)==2 for rows in units.values())
 diffs=lambda a,b:[sum(x[a]['correct']-x[b]['correct'] for x in v)/8 for v in units.values()]
 overall={a:{k:sum(row['scores'][a][k] for row in tests.values()) for k in ('correct','danger','tokens')} for a in sorted(expected_keys)}
 by_role={r:{a:sum(t['scores'][a]['correct'] for t in tests.values() if t['role']==r) for a in sorted(expected_keys)} for r in ROLES}
 delivered=interval(diffs('bound16','empty'));gap=interval(diffs('full','bound16'))
 assert len(clock['warm'])==720 and {x['row'] for x in clock['warm']}==ids
 def p95(x):return sorted(x)[(95*len(x)+99)//100-1]/1e6
 p_lookup=p95([r['lookup_ns'] for r in clock['warm']])
 complete=[]
 for repeat in range(5):
  for row in obs:
   t0=time.perf_counter_ns()
   f=row['features'];z=q[row['id']]
   fs=occupancy(f);sel=cache[key(f['role'],fs)]['16'];dec=parse(sel,16)
   for private in z['contexts']:
    assert valid_transition(receive(z['role'],private,dec),z)
   complete.append(time.perf_counter_ns()-t0)
   assert sel==m[row['id']]['bound16']
 p_combined=p95(complete)
 cost_practical=p_lookup<=.25 and p_combined<=1
 relevance=delivered[0]>.10 and gap[0]>.05
 safe=all(x['danger']==0 for x in overall.values())
 write('eligibility.json',{'scope':'constructed_only_not_qualified_no_fit','groups':len(units),'total_receiver_context_cases':576,'correctness':overall,'role_success_counts':by_role,'bounded_vs_empty_ci':delivered,'full_vs_bounded_ci':gap,'lookup_p95_ms':p_lookup,'complete_warm_p95_ms':p_combined,'compile_ms':clock['compile_ns']/1e6,'cache_bytes_json':clock['cache_bytes_json'],'cache_entries':clock['cache_entries'],'material_gap':relevance,'safety_guard':safe,'decision':'STOP' if cost_practical or not relevance or not safe else 'REVIEW_ONLY_NEVER_AUTO_FIT'})
 write('complete-clock.json',{'complete_ns':complete})
 write('metrics.json',{'groups':len(units)})
if __name__=='__main__':
 if sys.argv[1:] not in (['emit'],['verify']):raise ValueError('explicit stage')
 (emit if sys.argv[1]=='emit' else verify)()
