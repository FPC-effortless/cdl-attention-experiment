import os,sys,json,hashlib,time,tempfile
from pathlib import Path
from collections import defaultdict,Counter
import joblib,numpy as np
from scipy.sparse import hstack
from runtime import Runtime,EMPTY,HOLD,UNKNOWN
if sys.flags.optimize: raise RuntimeError('Optimized Python disables research integrity checks')
HERE=Path(__file__).resolve().parent
BASE=Path(os.environ['COG_BASE']);DATA=Path(os.environ['COG_DATA']);OUT=Path(os.environ.get('TRAINLAB_OUTPUT_DIR',HERE/'output'));OUT.mkdir(exist_ok=True,parents=True)
ARMS=['verified','unrestricted','soft','expiry2','expiry4','noop','reset']
def save(name,x): (OUT/name).write_text(json.dumps(x,sort_keys=True,allow_nan=False))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
 pins=json.loads((DATA/'pins.json').read_text()); rows=[]
 for file,h in pins.items():
  assert file.startswith('dev/') and digest(DATA/file)==h
  for d in json.loads((DATA/file).read_text()):
   history=[];old=[EMPTY,EMPTY]
   for t in d['turns']:
    text=t['utterance'].lower()
    if t['speaker']=='USER':
     labels={k:v for f in t['frames'] for k,v in f['state']['slot_values'].items()};truth=[labels.get(k,[EMPTY])[0] for k in ('hotel-area','hotel-pricerange')];delta=[v if v!=o else HOLD for v,o in zip(truth,old)]
     rows.append({'id':d['dialogue_id'],'turn':int(t['turn_id']),'text':[text,' '.join(history[-2:])],'truth':truth,'prior':old,'delta':delta});old=truth
    history.append(text)
 manifest=dict(l.split('  ')[::-1] for l in (BASE/'SHA256SUMS').read_text().splitlines());cp=BASE/'checkpoints/dialogue/models.joblib';assert digest(cp)==manifest['checkpoints/dialogue/models.joblib']
 assert digest(BASE/'results/dialogue-transfer.json')==manifest['results/dialogue-transfer.json']
 frozen=json.loads(Path(os.environ['FORGE_VIEW']).read_text());assert digest(Path(os.environ['FORGE_VIEW']))==os.environ['FORGE_SHA256'];assert rows==[x['features']['row'] for x in frozen['rows']];rows=[x['features']['row'] for x in frozen['rows']]
 vectors,models=joblib.load(cp);models=[models[k][0] for k in ('hotel-area','hotel-pricerange')];names=['hotel-area','hotel-pricerange'];labels={k:m.classes_.tolist() for k,m in zip(names,models)}
 meta=json.loads((BASE/'results/dialogue-transfer.json').read_text());gates={k:meta['slots'][k+'/direct']['gated']['threshold'] for k in names};binding=hashlib.sha256(''.join(digest(p) for p in [cp,BASE/'results/dialogue-transfer.json',HERE/'PROTOCOL.md',HERE/'runtime.py',HERE/'upstream/belief.py',HERE/'upstream/state_store.py']).encode()).hexdigest()
 return rows,vectors,models,labels,gates,binding

def infer(texts,vectors,models):
 x=hstack([v.transform([t[field] for t in texts]) for field,v in vectors],format='csr');return [m.predict_proba(x) for m in models]
def score(trace):
 by=defaultdict(Counter);episodes=[];groups=defaultdict(list)
 for r in trace:groups[r['id']].append(r)
 for id,rs in groups.items():
  c=by[id];runs=[0,0];priorread=None
  for r in rs:
   pred=r['read'];truth=r['truth'];changed=any(x!=HOLD for x in r['delta']);ok=pred==truth
   c.update(turns=1,incorrect=int(not ok),joint=int(ok),changed=int(changed),changed_joint=int(ok and changed),unchanged=int(not changed),unchanged_joint=int(not changed and ok),unchanged_wrong=int(not changed and not ok),slot_correct=sum(a==b for a,b in zip(pred,truth)),abstain=sum(a==UNKNOWN for a in pred),false_assertion=sum(a!=b and a!=UNKNOWN for a,b in zip(pred,truth)))
   for j,(a,b,d,p) in enumerate(zip(pred,truth,r['delta'],r['prior'])):
    kind='hold' if d==HOLD else 'clear' if d==EMPTY else 'first_set' if p==EMPTY else 'overwrite';c[kind+'_n']+=1;c[kind+'_correct']+=int(a==b)
    if priorread is not None and d==HOLD and priorread[j]==b:
     c['mutation_eligible']+=1
     c['false_mutation']+=int(a!=b and a!=UNKNOWN);c['abstain_mutation']+=int(a==UNKNOWN)
   for k,bad in enumerate([not ok,any(a!=b and a!=UNKNOWN for a,b in zip(pred,truth))]):
    runs[k]=runs[k]+1 if bad else 0;c['longest_'+['incorrect','false_assertion'][k]]=max(c['longest_'+['incorrect','false_assertion'][k]],runs[k])
   priorread=pred
  c['end_incorrect_run']=runs[0];c['end_false_assertion_run']=runs[1]
  for slot in range(2):
   starts=[i for i,r in enumerate(rs) if r['delta'][slot]!=HOLD]
   for a,b in zip(starts,starts[1:]+[len(rs)]):
    good=[i-a for i in range(a,b) if rs[i]['read'][slot]==rs[i]['truth'][slot]]
    episodes.append({'id':id,'slot':slot,'start':rs[a]['turn'],'duration':b-a,'delay':good[0] if good else None,'censored':not good,'censor_reason':('next_change' if b<len(rs) else 'dialogue_end') if not good else None,'error_debt':sum(rs[i]['read'][slot]!=rs[i]['truth'][slot] for i in range(a,b))})
 total=Counter()
 for c in by.values():total.update(c)
 total['episodes']=len(episodes);total['never_correct']=sum(e['censored'] for e in episodes);total['error_debt']=sum(e['error_debt'] for e in episodes)
 return {'total':dict(total),'dialogues':dict(by),'episodes':episodes}

def main():
 rows,vectors,models,labels,gates,binding=load();scores=infer([r['text'] for r in rows],vectors,models);grouped=defaultdict(list)
 for i,r in enumerate(rows):grouped[r['id']].append((r,[p[i] for p in scores]))
 coverage=Counter()
 for i,r in enumerate(rows):
  for j,d in enumerate(r['delta']):
   if d!=HOLD:
    coverage['true_changes']+=1;ls=list(labels.values())[j]
    for k in (1,2,3):coverage['top'+str(k)]+=int(d in [ls[z] for z in np.argsort(-scores[j][i],kind='stable')[:k]])
 traces={};metrics={};checks=Counter()
 with tempfile.TemporaryDirectory() as tmp:
  for arm in ARMS:
   trace=[]
   for id,items in grouped.items():
    r=Runtime(labels,gates,arm,binding);restored=None
    for n,(row,p) in enumerate(items):
     output=r.observe(row['turn'],p)
     if restored:
      assert restored.observe(row['turn'],p)==output;assert restored.dump()==r.dump();checks['matched_restored_turns']+=1
     if n==len(items)//2:
      path=Path(tmp)/'snapshot.json';r.save(path);restored=Runtime.restore(path,binding,arm);assert restored.dump()==r.dump();checks['snapshots']+=1
      original=path.read_text();bad=json.loads(original);bad['body']['last']+=1;path.write_text(json.dumps(bad))
      try:Runtime.restore(path,binding,arm);raise AssertionError('corruption accepted')
      except ValueError:checks['corruption_rejected']+=1
      path.write_text(original)
     trace.append({k:row[k] for k in ('id','turn','truth','prior','delta')}|output)
    try:r.observe(items[-1][0]['turn'],items[-1][1]);raise AssertionError('duplicate accepted')
    except ValueError:checks['duplicates_rejected']+=1
    r.close()
    if restored:restored.close()
   traces[arm]=trace;metrics[arm]=score(trace)
  for arm,trace in traces.items():
   assert [(r['patch'],r['verified']) for r in trace]==[(r['patch'],r['verified']) for r in traces['verified']]
 save('traces.json',traces);save('metrics_detail.json',metrics)
 ids=sorted(grouped);comparison={}
 for arm in ARMS:
  comparison[arm]={}
  for base in ['unrestricted','soft']:
   comparison[arm][base]={}
   for metric in ['unchanged_wrong','false_assertion','abstain','joint']:
    delta=np.array([metrics[arm]['dialogues'][i][metric]-metrics[base]['dialogues'][i][metric] for i in ids]);rng=np.random.default_rng(62064);draw=rng.choice(delta,(5000,len(ids)),replace=True).sum(1)
    comparison[arm][base][metric]={'candidate_minus_baseline':int(delta.sum()),'ci':np.quantile(draw,[.025,.975]).tolist()}
 save('results.json',{'scope':'previously_exposed_DEV_diagnostic_NOT_QUALIFIED','rows':len(rows),'dialogues':len(ids),'coverage':dict(coverage),'totals':{k:v['total'] for k,v in metrics.items()},'comparisons':comparison,'recovery':dict(checks),'binding':binding})
 save('metrics.json',{'turns':len(rows)})
 print(json.dumps({'rows':len(rows),'coverage':dict(coverage),'recovery':dict(checks)}))
def timing():
 rows,vectors,models,labels,gates,binding=load();grouped=defaultdict(list)
 for r in rows:grouped[r['id']].append(r)
 measurements=[]
 for rep in range(3):
  for arm in ARMS[rep:]+ARMS[:rep]:
   start=time.perf_counter();turns=0
   for id,rs in grouped.items():
    r=Runtime(labels,gates,arm,binding)
    for row in rs:
     p=infer([row['text']],vectors,models);r.observe(row['turn'],[x[0] for x in p]);turns+=1
    r.close()
   measurements.append({'rep':rep,'arm':arm,'turns':turns,'elapsed_seconds':time.perf_counter()-start});save('timing.json',measurements);print('timing',rep,arm,measurements[-1]['elapsed_seconds'],flush=True)
 recovery=[]
 with tempfile.TemporaryDirectory() as tmp:
  for arm in ARMS:
   r=Runtime(labels,gates,arm,binding)
   first=next(iter(grouped.values()))
   for row in first[:max(1,len(first)//2)]:
    p=infer([row['text']],vectors,models);r.observe(row['turn'],[x[0] for x in p])
   for rep in range(20):
    path=Path(tmp)/'snapshot.json';a=time.perf_counter();r.save(path);b=time.perf_counter();copy=Runtime.restore(path,binding,arm);c=time.perf_counter();assert copy.dump()==r.dump();copy.close();recovery.append({'arm':arm,'rep':rep,'snapshot_seconds':b-a,'restore_seconds':c-b,'bytes':path.stat().st_size})
   r.close()
 save('recovery_timing.json',recovery)
 save('metrics.json',{'timed_sessions':len(measurements)})
if __name__=='__main__': timing() if len(sys.argv)>1 and sys.argv[1]=='timing' else main()
