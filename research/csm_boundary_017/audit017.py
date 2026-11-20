"""Independent saved-record scoring against frozen case gold, not claimed metrics.
No evaluator imports. Finite development panels only; no TEST access.
"""
import math,random,statistics
FIELDS=('query_accuracy','final_map_accuracy','final_state_accuracy','complete_session_accuracy','exact_event_precision','exact_event_recall')
def action(e):
 k=e['op'];return [e['pos'],k,e['dst'] if k else 0,e['src'] if k in (2,3) else 0,e['value'] if k==1 else 0]
def positions(xs,n):
 if type(xs)!=list or any(type(x)!=int or not 0<=x<n for x in xs) or xs!=sorted(set(xs)):raise ValueError('invalid positions')
def state(s):
 if type(s)!=list or len(s)!=4:raise ValueError('state shape')
 for cell in s:
  if type(cell)!=list or len(cell)!=2:raise ValueError('cell shape')
  k,v=cell
  if type(k)!=int or type(v)!=int or not ((k==0 and v==0) or (k==1 and 1<=v<=4) or (k==2 and 0<=v<4)):raise ValueError('invalid cell')
def rescore(records,cases,expected_arm=None):
 ids=[c['case_id'] for c in cases]
 if not ids or len(ids)!=len(set(ids)) or [r['case_id'] for r in records]!=ids:raise ValueError('case roster mismatch')
 totals=dict(cases=len(cases),queries=0,correct=0,maps=0,states=0,sessions=0,exact=0,pred=0,gold=0,raw_commits=0,raw_interior=0,suppressed=0);per_case=[]
 for r,c in zip(records,cases):
  if r.get('family')!=c['family']:raise ValueError('record family mismatch')
  if expected_arm is not None and r.get('arm')!=expected_arm:raise ValueError('record arm mismatch')
  n=len(c['text'].encode('ascii'));ev=c['events'];gold=[action(e) for e in ev if e['op']];q=[e for e in ev if e['op']==5]
  expected={'length':n,'gold_nonnoop':gold,'query_positions':[e['pos'] for e in q],'query_targets':[e['answer'] for e in q],'final_query_positions':c['final_query_positions'],'final_query_targets':[e['answer'] for e in q][-4:],'terminal_positions':[e['pos'] for e in ev],'gold_terminal_states':[e['state_after'] for e in ev],'gold_final_state':c['final_state']}
  if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('gold evidence differs from frozen data')
  pred=r['pred_nonnoop'];positions([e[0] for e in pred],n)
  for e in pred:
   if len(e)!=5 or any(type(v)!=int for v in e):raise ValueError('action schema')
   p,k,d,s,v=e
   if not 1<=k<=5 or not 0<=d<4 or (k in (2,3) and not 0<=s<4) or (k not in (2,3) and s!=0) or (k==1 and not 1<=v<=4) or (k!=1 and v!=0):raise ValueError('action values')
  yp=r['query_predictions'];fp=r['final_query_predictions']
  if len(yp)!=len(q) or len(fp)!=4 or fp!=yp[-4:] or any(type(v)!=int or not -1<=v<=4 for v in yp):raise ValueError('query schema')
  pts=r['pred_terminal_states']
  if len(pts)!=len(ev):raise ValueError('terminal state count')
  for s in pts+[r['pred_final_state']]:state(s)
  raw=r['raw_nonnoop_positions'];interior=r['raw_interior_commits'];supp=r['suppressed_commits']
  for x in (raw,interior,supp):positions(x,n)
  if not {e[0] for e in pred}.issubset(set(raw)):raise ValueError('executed event absent from raw policy')
  if interior!=[p for p in raw if ord(c['text'][p]) not in (46,63)] or supp!=[p for p in raw if p not in {e[0] for e in pred}]:raise ValueError('raw diagnostics inconsistent')
  correct=sum(a==e['answer'] for a,e in zip(yp,q));mp=fp==[e['answer'] for e in q][-4:];st=r['pred_final_state']==c['final_state'];complete=pred==gold and correct==len(q) and pts==[e['state_after'] for e in ev] and st
  for key,v in dict(queries=len(q),correct=correct,maps=int(mp),states=int(st),sessions=int(complete),exact=len({tuple(e) for e in gold}&{tuple(e) for e in pred}),pred=len(pred),gold=len(gold),raw_commits=len(raw),raw_interior=len(interior),suppressed=len(supp)).items():totals[key]+=v
  per_case.append({'case_id':c['case_id'],'family':c['family'],'complete_session':int(complete),'complete_map':int(mp)})
 ratio=lambda a,b:totals[a]/totals[b] if totals[b] else None
 metrics=dict(zip(FIELDS,[ratio('correct','queries'),ratio('maps','cases'),ratio('states','cases'),ratio('sessions','cases'),ratio('exact','pred'),ratio('exact','gold')]))
 return {'metrics':metrics,'counts':totals,'per_case':per_case,'qualified':all(v is not None and v>=.95 for v in metrics.values())}
def audit_result(result,data,arm,seed):
 if result.get('arm')!=arm or result.get('seed')!=seed:raise ValueError('arm/seed mismatch')
 out={}
 for split,key in [('train_probe','train'),('dev','dev')]:
  if set(result.get(split,{}))!=set(data[key]):raise ValueError('panel roster mismatch')
  out[split]={p:rescore(result[split][p]['records'],cases,expected_arm=arm) for p,cases in data[key].items()}
 return {'arm':arm,'seed':seed,'panels':out,'qualified':all(p['qualified'] for s in out.values() for p in s.values())}
def audit_study(results,data,arms,seeds=(101,202,303)):
 """Require every prespecified arm/seed, retain failures, never pool to qualify."""
 expected={(a,s) for a in arms for s in seeds}
 if not expected or len(expected)!=len(arms)*len(seeds) or len(results)!=len(expected):raise ValueError('study roster mismatch')
 keys=[(r.get('arm'),r.get('seed')) for r in results]
 if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('study arm/seed mismatch')
 audited=[audit_result(r,data,r['arm'],r['seed']) for r in results]
 return {'runs':audited,'qualified_by_arm':{a:all(r['qualified'] for r in audited if r['arm']==a) for a in arms}}
def paired_uncertainty(left,right,seeds=(101,202,303),repeats=2000,bootstrap_seed=17017):
 """Crossed paired seed/family bootstrap of complete-session accuracy differences.
 Inputs seed -> per_case rows. Family clustering conservatively keeps payload,
 binding and renderer variants together. Finite-panel descriptive uncertainty,
 NOT broad-generalization confidence; only three seeds/few families.
 """
 if set(left)!=set(seeds) or set(right)!=set(seeds) or repeats<100:raise ValueError('seed roster/repeats')
 table={};families=None;case_roster=None
 for seed in seeds:
  a=left[seed];b=right[seed]
  if case_roster is None:case_roster=[r['case_id'] for r in a]
  if [r['case_id'] for r in a]!=case_roster:raise ValueError('cross-seed case roster mismatch')
  if not a or len({r['case_id'] for r in a})!=len(a) or [r['case_id'] for r in a]!=[r['case_id'] for r in b]:raise ValueError('paired roster mismatch')
  groups={}
  for x,y in zip(a,b):
   if x['family']!=y['family'] or type(x['complete_session']) is not int or type(y['complete_session']) is not int or x['complete_session'] not in (0,1) or y['complete_session'] not in (0,1):raise ValueError('invalid paired values')
   groups.setdefault(x['family'],[]).append(x['complete_session']-y['complete_session'])
  if families is None:families=sorted(groups)
  if set(groups)!=set(families):raise ValueError('family mismatch')
  table[seed]={g:statistics.mean(v) for g,v in groups.items()}
 rng=random.Random(bootstrap_seed);draw=[]
 for _ in range(repeats):
  ss=rng.choices(list(seeds),k=len(seeds));gg=rng.choices(families,k=len(families));draw.append(statistics.mean(table[s][g] for s in ss for g in gg))
 draw.sort();point=statistics.mean(table[s][g] for s in seeds for g in families)
 return {'estimand':'equally weighted seed/family mean paired complete-session difference, left minus right','difference':point,'percentile_95':[draw[int(.025*(repeats-1))],draw[int(.975*(repeats-1))]],'seed_family_differences':table,'seeds':list(seeds),'families':families,'bootstrap_seed':bootstrap_seed,'repeats':repeats,'limitation':'descriptive finite-development-panel interval; family variants and queries not independent; small seed/family counts'}
