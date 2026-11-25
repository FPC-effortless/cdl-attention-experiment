"""Weight-free scoring for sealed 120-family and remap contrast actors."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,math,random,sys
from statistics import mean
from trainlab.runner import verify_run
ROOT=Path(__file__).resolve().parent
AS={'ordinary':'d74d4659d487806e74dce30e519d57a53fb39129c0d7ba46874e41fad404a9e9','contrast':'a0369773833feb97349dac595bd176c2eb6f6fd6cb132b3251e15d329bd2c0c1'}
JS={'ordinary':'4b36c29d4dc1966163bddffde98d4e3e88b504455630c66532833c06cbb01459','contrast':'e21a40fe1c67071caa6981ca646349b3829847495478a338dd57026b37e550f7'}
SEEDS=(1771,1871,1971);ARMS=('consistency000','consistency025')
def req(x,m):
 if not x:raise ValueError(m)
def read(p,pin=None):
 b=p.read_bytes();req(pin is None or sha256(b).hexdigest()==pin,f'pin {p}')
 return json.loads(b)
def h(p):return sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,allow_nan=False,indent=2,sort_keys=True);f.write('\n')
 print(h(p))
def roster():
 items={(a,s,e,'ordinary') for a in ARMS for s in SEEDS for e in (128,1280)}
 items|={(a,s,1280,'contrast') for a in ARMS for s in SEEDS}
 return {f'{a}-{s}-{e}-{p}' for a,s,e,p in items|{('native',1371,0,'ordinary'),('native',1371,0,'contrast')}}
def seal(ad,runs,lp,out):
 led=read(ad/'ledger.json',lp); result={};req(set(led)==roster(),'actor registry')
 for label,c in led.items():
  d=ad/label;r=read(d/'cli-output.json');run=runs/r['id']
  req(verify_run(run)['valid'] and r==read(run/'run.json') and r['status']=='succeeded'
      and r['inputs']==c['inputs'] and r['name']==label and r['seed']==c['seed'] and r['elapsed_seconds']<=30,
      'native actor exact condition')
  req(h(d/'recipe.json')==c['recipe_sha256'] and read(d/'recipe.json')==read(run/'experiment.json')
      and all(h(d/k)==v for k,v in c['inputs'].items()),'actor custody')
  n=960 if c['panel']=='ordinary' else 768
  req(r['metrics']==dict(prediction_rows=n,model_forwards=n//8,optimizer_updates=0)
      and set(r['artifacts'])=={'predictions.json'},'actor scope')
  result[label]=dict(id=r['id'],receipt_sha256=h(run/'run.json'),prediction_sha256=r['artifacts']['predictions.json'])
 req(len({x['id'] for x in result.values()})==20,'duplicate actors')
 write(out,dict(ledger_sha256=lp,results=result))

def ranks(values,y,z):
 req(len(values)==287 and all(math.isfinite(v) for v in values),'rank geometry')
 x=values[y];full=all(x>v for i,v in enumerate(values) if i!=y);rel=x>values[z]
 req(not full or rel,'rank implication')
 return full,rel
def reconstruct(prompts,logits,vocab):
 idx={k:i+262 for i,k in enumerate(vocab)}; result=[]
 req(len(idx)==25 and len(logits)==len(prompts),'vocab or logits coverage')
 for p,o in zip(prompts,logits,strict=True):
  a,eq,x,b,eq2,y,q=p.split();req(eq==eq2=='is' and a!=b and x!=y and q in (a,b),'grammar')
  t,z,first=(x,y,True) if q==a else (y,x,False)
  hi,rel=ranks(o,idx[t],idx[z]);result.append((t,z,first,hi,rel))
 return result
def ordinary(j,rows):
 g={k:Counter() for k in ('all','seen_name_pair','unseen_name_pair')}
 req(len(j)==120 and len(rows)==960,'ordinary coverage')
 for i,f in enumerate(j):
  rr=list(range(8*i,8*i+8));req(f['rows']==rr,'row order')
  z=[rows[k] for k in rr];req([x[0] for x in z]==f['answers'],'ground truth')
  w=[x[3] for x in z];rel=[x[4] for x in z]
  counts=dict(families=1,worlds=8,strict_worlds=sum(w),strict_families=all(w),relative_worlds=sum(rel),
      relative_families=all(rel),first_strict=sum(x[2] and x[3] for x in z),second_strict=sum(not x[2] and x[3] for x in z),
      both_queries=sum(w[k] and w[k+1] for k in (0,2,4,6)),
      both_orders=sum(w[k] and w[k+2] for k in (0,1,4,5)),
      both_assignments=sum(w[k] and w[k+4] for k in range(4)))
  g['all'].update(counts);g['seen_name_pair' if f['familiar_name_pair'] else 'unseen_name_pair'].update(counts)
 return g
def boot(d):
 rr=random.Random(20261130)
 xs=sorted(mean(rr.choices(d,k=len(d))) for _ in range(2000))
 return [xs[50],xs[1949]]
def contrast(j,q):
 out=defaultdict(Counter); blocks=defaultdict(list);used=set()
 req(len(j)==24 and len(q)==768,'contrast coverage')
 for b in j:
  raw=defaultdict(list)
  for c in b['cells']:
   lab=f"{c['queried']}_{c['focal_position']}";hits={}
   req(set(c['alternatives'])=={'F0','F1','R0','R1'},'contrast alternatives')
   for role,i in c['alternatives'].items():
    req(i not in used,'duplicate row');used.add(i)
    row=q[i]; req((row[0],row[1])==(c['answer'],c['foil']),'ground truth')
    group='fragile' if role[0]=='F' else 'robust'
    out[lab][group+'_strict']+=row[3];out[lab][group+'_relative']+=row[4];out[lab][group+'_worlds']+=1
    hits[role]=row[3]
   raw[lab].append((hits['R0']+hits['R1']-hits['F0']-hits['F1'])/2)
  for k,v in raw.items():req(len(v)==2,'missing assignments');blocks[k].append(mean(v))
 req(used==set(range(768)) and all(len(v)==24 for v in blocks.values()),'contrast rows lost')
 return dict(counts=out,robust_minus_fragile={k:dict(effect=mean(v),bootstrap_95=boot(v),block_values=v) for k,v in blocks.items()})
def score(ad,runs,lp,seal,sp,out):
 z=read(seal,sp); led=read(ad/'ledger.json',lp)
 req(z['ledger_sha256']==lp and set(led)==set(z['results'])==roster(),'sealed actor registry')
 panel={k:(read(ROOT/f'eval_panels/{k}_actor.json',AS[k])['prompts'],read(ROOT/f'eval_panels/{k}_judge.json',JS[k])) for k in AS}
 observed={}; vocab=None
 for label,c in led.items():
  s=z['results'][label];d=runs/s['id'];r=read(d/'run.json',s['receipt_sha256']);p=read(d/'output/predictions.json',s['prediction_sha256'])
  req(r['artifacts']['predictions.json']==s['prediction_sha256'] and
      (p['arm'],p['seed'],p['updates'],p['backbone_sha256'],p['adapter_sha256'],p['actor_sha256'])==
      (c['arm'],c['seed'],c['endpoint'],c['backbone_sha256'],c['adapter_sha256'],AS[c['panel']]),'actor identity')
  v=p['tokenizer_words'];vocab=v if vocab is None else vocab
  req(v==vocab and p['vocab_size']==287,'vocabulary identity')
  a,j=panel[c['panel']];q=reconstruct(a,p['logits'],v)
  observed[label]=ordinary(j['families'],q) if c['panel']=='ordinary' else contrast(j['blocks'],q)
 n=observed['native-1371-0-ordinary']['all'];dec={}
 for seed in SEEDS:
  lit,con=[observed[f'{a}-{seed}-1280-ordinary']['all'] for a in ARMS]
  lc,cc=[observed[f'{a}-{seed}-1280-contrast']['counts']['focal_second'] for a in ARMS]
  native_first=n['first_strict'];floor=lambda z:z['strict_worlds']>=912 and z['strict_families']>=108
  dec[seed]=dict(zero_clears=floor(lit),consistent_clears=floor(con),consistent_first_delta=con['first_strict']-native_first,
    zero_first_delta=lit['first_strict']-native_first,both_query_delta=con['both_queries']-lit['both_queries'],
    fragile_second_delta=cc['fragile_strict']-lc['fragile_strict'],robust_second_delta=cc['robust_strict']-lc['robust_strict'])
 ok=all(d['consistent_clears'] and d['consistent_first_delta']>=-5 and
        d['fragile_second_delta']>0 and d['robust_second_delta']>=0 for d in dec.values())
 write(out,dict(scores=observed,per_seed=dec,consistency_preferred_on_known_backbone=ok,broader_model_qualification=False))
if __name__=='__main__':
 op,ad,ru,l,*a=sys.argv[1:]
 if op=='seal':seal(Path(ad),Path(ru),l,Path(a[0]))
 elif op=='score':score(Path(ad),Path(ru),l,Path(a[0]),a[1],Path(a[2]))
 else:raise ValueError('bad stage')
