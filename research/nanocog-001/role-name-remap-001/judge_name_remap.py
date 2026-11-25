"""Read-only exact name-remap scorer. Pure Python, no model invocation."""
from collections import Counter,defaultdict
from hashlib import sha256
import itertools,json,math,random,sys
from pathlib import Path
from statistics import mean
from trainlab.runner import verify_run
A='0775d4358384f12486d616988aefa8a764888ece48e8942f96c617754ff07514'
J='3e1eecfc644f11bad7c722cce6280a39f2650fe93d750a1a9fb52a0bc7d42947'
BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
ADP='abfd950297ca1b90846621f770601b7e38081f5c62eea49e4e1c814477674932'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
 b=p.read_bytes();need(pin is None or sha256(b).hexdigest()==pin,'bad pin '+str(p))
 return json.loads(b)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,allow_nan=False,indent=2,sort_keys=True);f.write('\n')
 print(h(p))
def seal(ad,runs,ledger_pin,out):
 led=read(ad/'ledger.json',ledger_pin);o={};need(len(led)==2,'roster')
 for lab,c in led.items():
  box=ad/lab;r=read(box/'cli-output.json'); native=runs/r['id']
  need(verify_run(native)['valid'] and r==read(native/'run.json') and r['status']=='succeeded' and r['inputs']==c['inputs'] and r['name']==lab and 0<r['elapsed_seconds']<=45,'native')
  need(h(box/'recipe.json')==c['recipe_sha256'] and all(h(box/z)==q for z,q in c['inputs'].items()),'closure')
  need(r['metrics']==dict(prediction_rows=768,model_forwards=96,optimizer_updates=0)
       and set(r['artifacts'])=={'predictions.json'},'scope')
  o[lab]=dict(id=r['id'],run_sha256=h(native/'run.json'),predictions_sha256=r['artifacts']['predictions.json'])
 need(len({x['id'] for x in o.values()})==2,'duplicate')
 write(out,dict(ledger_sha256=ledger_pin,runs=o))

def bootstrap(x):
 r=random.Random(20261128);n=len(x)
 v=sorted(mean(r.choices(x,k=n)) for _ in range(2000))
 return (v[50],v[1949])
def strict(logits,a,b):
 need(len(logits)==287 and all(math.isfinite(y) for y in logits),'bad logits')
 p=logits[a]
 return all(p>y for i,y in enumerate(logits) if i!=a),p>logits[b]
def flip_class(old,new):
 return 'rescued' if new and not old else 'damaged' if old and not new else 'both_correct' if old else 'both_wrong'
def score_model(p,actor,blocks):
 lookup={a:i+262 for i,a in enumerate(p['tokenizer_words'])}
 need(len(lookup)==25 and p['vocab_size']==287 and len(p['logits'])==768,'vocab/outputs')
 stricts=defaultdict(Counter);relative=defaultdict(Counter);paired=defaultdict(Counter)
 names=defaultdict(Counter); block_deltas=defaultdict(list);allrows=set()
 for b in blocks:
  diff=defaultdict(list)
  for c in b['cells']:
   typ=f"{c['queried']}_{c['focal_position']}"
   hits={}; vmap={}
   for role,i in c['alternatives'].items():
    need(i not in allrows,'dup score row');allrows.add(i)
    z=actor[i].split(); fr,iso,k,sr,iss,y,ask=z
    who=b['roles'][role]
    which=fr if c['focal_position']=='first' else sr
    need(iso==iss=='is' and which==who and ask==(who if c['queried']=='focal' else b['distractor']),'frozen roles')
    ans,foil=(k,y) if ask==fr else (y,k)
    need(c['answer']==ans and c['foil']==foil,'independent truth')
    x,rel=strict(p['logits'][i],lookup[ans],lookup[foil]);hits[role]=x;vmap[role]=z
    category='fragile' if role.startswith('F') else 'robust'
    for out,yes in ((stricts,x),(relative,rel)):
     out[typ][f'{category}_correct']+=yes;out[typ][f'{category}_total']+=1
    names[who][typ+'_correct']+=x;names[who][typ+'_total']+=1
   need(set(hits)=={'F0','F1','R0','R1'},'cell alternatives')
   for i,j in itertools.combinations(vmap,2):
    dose=sum(x!=y for x,y in zip(vmap[i],vmap[j]))
    need(dose==(2 if c['queried']=='focal' else 1),'substitution dose changed')
   diff[typ].append((hits['R0']+hits['R1']-hits['F0']-hits['F1'])/2)
   for f,r in (('F0','R0'),('F1','R1')):
    paired[typ]['cross_'+flip_class(hits[f],hits[r])]+=1
   paired[typ]['within_fragile_disagree']+=hits['F0']!=hits['F1']
   paired[typ]['within_robust_disagree']+=hits['R0']!=hits['R1']
  for t,v in diff.items():
   need(len(v)==2,'block coverage');block_deltas[t].append(mean(v))
 need(allrows==set(range(768)) and all(len(v)==24 for v in block_deltas.values()),'panel coverage')
 return dict(strict_by_stratum=stricts,two_color_by_stratum=relative,paired=paired,by_focal_name=names,
     robust_minus_fragile={k:dict(effect=mean(v),bootstrap_95=bootstrap(v),blocks=v) for k,v in block_deltas.items()})
def score(ad,runs,pin,seal_path,seal_pin,out):
 led=read(ad/'ledger.json',pin);s=read(seal_path,seal_pin)
 need(s['ledger_sha256']==pin and set(s['runs'])==set(led) and len(led)==2,'sealed roster')
 actor=read(Path('panel/actor.json'),A)['prompts'];block=read(Path('panel/judge.json'),J)['blocks']
 need(len(actor)==len(set(actor))==768 and len(block)==24,'bad panel')
 result={}
 for lab,sg in s['runs'].items():
  d=runs/sg['id']; rr=read(d/'run.json',sg['run_sha256'])
  p=read(d/'output/predictions.json',sg['predictions_sha256'])
  need(rr['artifacts']['predictions.json']==sg['predictions_sha256'] and
       p['actor_sha256']==A and p['backbone_sha256']==BASE and
       ((p['arm'],p['seed'],p['updates'],p['adapter_sha256'])==
        (('native',1371,0,'none') if lab=='remap-native-1371' else ('tap_compare',1471,1280,ADP))),'wrong exact bindings')
  result[lab]=score_model(p,actor,block)
 need(len(result)==2,'scope')
 common=all(x['robust_minus_fragile']['focal_second']['bootstrap_95'][0]>0 for x in result.values())
 write(out,dict(results=result,common_recurrence_under_tested_substitutions=common,qualified=False))
if __name__=='__main__':
 op,ad,runs,pin,*a=sys.argv[1:]
 if op=='seal':seal(Path(ad),Path(runs),pin,Path(a[0]))
 elif op=='score':score(Path(ad),Path(runs),pin,Path(a[0]),a[1],Path(a[2]))
 else:raise ValueError('unknown operation')
