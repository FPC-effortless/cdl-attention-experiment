"""Verify/seal native probe outputs and then score only retained logits."""
from collections import Counter
from hashlib import sha256
import json,math,random,sys
from pathlib import Path
from statistics import mean,median
from trainlab.runner import verify_run
A='44f380f0e934450e3c809162578381e670683602a4e52a4487c89fd2197a8116'
J='d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848'
L='122dbb6f4417a39d1acf32e970b45ae8722657e7b09dc54c375ea9aa809ac612'
def check(x,msg):
 if not x: raise ValueError(msg)
def blob(p): return p.read_bytes()
def h(p):return sha256(blob(p)).hexdigest()
def read(p,d=None):
 b=blob(p);check(d is None or sha256(b).hexdigest()==d,'digest')
 return json.loads(b)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,allow_nan=False,sort_keys=True,indent=2)
P=Path('probe_admissions')
R=Path('probe_runs')
def seal():
 out={}
 for k,a in read(P/'ledger.json',L).items():
  rr=read(P/k/'cli-output.json')
  p=R/rr['id']
  check(verify_run(p)['valid'] and rr==read(p/'run.json') and rr['status']=='succeeded' and rr['inputs']==a['inputs'],'native actor')
  check(h(P/k/'recipe.json')==a['recipe_sha256'] and rr['name']==k and rr['seed']==a['seed'],'actor registry')
  check(rr['metrics']==dict(prediction_rows=256,model_forwards=160,optimizer_updates=0),'native scope')
  out[k]=dict(id=rr['id'],receipt_sha256=h(p/'run.json'),predictions_sha256=rr['artifacts']['predictions.json'])
 check(len(out)==2 and len({x['id'] for x in out.values()})==2,'actor coverage')
 s=Path('PROBE_SEAL.json');write(s,out);print(h(s))
def bootstrap(values):
 rng=random.Random(20261125)
 n=len(values); samples=sorted(mean(rng.choices(values,k=n)) for _ in range(2000))
 return [samples[50],samples[1949]]
def score(pin):
 led=read(P/'ledger.json',L);se=read(Path('PROBE_SEAL.json'),pin)
 check(set(se)==set(led),'registry')
 a=read(Path('probe_panel/actor.json'),A)['prompts']; j=read(Path('probe_panel/judge.json'),J)['families']
 check(len(a)==256 and len(j)==32,'panel shape')
 rows=[]
 for i,item in enumerate(j):
  for q,s in enumerate(a[8*i:8*i+8]):
   x,w,c,y,z,d,t=s.split();check(w==z=='is' and x!=y and c!=d and t in (x,y),'grammar')
   answer,other=(c,d) if t==x else (d,c);check(item['answers'][q]==answer,'truth')
   rows.append((answer,other,'first' if t==x else 'second'))
  check(all(a[8*i+q].split()[:6]==a[8*i+q+1].split()[:6] and rows[8*i+q][0]!=rows[8*i+q+1][0] for q in (0,2,4,6)),'query pairs')
 out={}
 for k,s in se.items():
  run=R/s['id']; rr=read(run/'run.json',s['receipt_sha256'])
  pkt=read(run/'output/predictions.json',s['predictions_sha256'])
  check(pkt['checkpoint_sha256']==led[k]['checkpoint_sha256'] and pkt['seed']==led[k]['seed'] and pkt['actor_sha256']==A,'exact bindings')
  words=pkt['tokenizer_words']; check(len(words)==len(set(words))==25,'vocab')
  ids={w:i+262 for i,w in enumerate(words)}
  conditions=pkt['logits'];check(set(conditions)=={'native','opposite','sham'},'branches')
  cnt={}; margins={}; correct={}
  for branch,logits in conditions.items():
   check(len(logits)==256,'logit cardinality')
   c=Counter(); m=[]; hits=[]
   for (y,f,side),v in zip(rows,logits,strict=True):
    check(len(v)==287 and all(math.isfinite(t) for t in v),'finite logits')
    yid=ids[y]; gap=v[yid]-v[ids[f]]
    exact=all(v[yid]>t for i,t in enumerate(v) if i!=yid)
    check(not exact or gap>0,'containment')
    c['strict_worlds']+=exact;c['relative_worlds']+=gap>0
    c[f'{side}_strict']+=exact;c[f'{side}_relative']+=gap>0
    m.append(gap);hits.append(exact)
   c['strict_both_queries']=sum(hits[i] and hits[i+1] for i in range(0,256,2))
   c['strict_complete_family']=sum(all(hits[i:i+8]) for i in range(0,256,8))
   c['never_changes_choice_pairs']=sum(m[i]*m[i+1]<0 for i in range(0,256,2))
   c['strict_rescues']=sum(x and not y for x,y in zip(hits,correct.get('native',hits)))
   c['strict_damages']=sum(y and not x for x,y in zip(hits,correct.get('native',hits)))
   c['mean_query_response']=mean(m)
   c['mean_first_slot_bias']=mean(x if r[2]=='first' else -x for x,r in zip(m,rows))
   cnt[branch]=c; margins[branch]=m;correct[branch]=hits
  m=margins
  delta=[mean(m['sham'][u]-m['opposite'][u] for u in range(i,i+8)) for i in range(0,256,8)]
  info=pkt['diagnostics'];check(all(len(v)==256 and all(math.isfinite(x) for x in v) for v in info.values()),'dose')
  out[k]=dict(branches=cnt, opposite_minus_sham_response_loss=mean(delta),family_bootstrap_95=bootstrap(delta),
             score_dose_median=median(info['score_l2']), zero_dose=sum(x<=1e-10 for x in info['score_l2']),
             collinear_sham=sum(x>=.99 for x in info['sham_abs_cos']), dose_prob_opposite_median=median(info['opposite_prob_l2']),
             dose_prob_sham_median=median(info['sham_prob_l2']))
 write(Path('PROBE_RESULTS.json'),out);print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':
 if sys.argv[1]=='seal':seal()
 elif sys.argv[1]=='score':score(sys.argv[2])
 else:raise ValueError('unknown task')
