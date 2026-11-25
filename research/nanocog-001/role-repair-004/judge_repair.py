"""Exact sealed standalone scores for frozen adapter validation (no model)."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json,math,sys
from trainlab.runner import verify_run

A='6915a754aa598d5608f0041850b3897ecfeef0bd8731835815e24333ba6c98d1'
J='0a3ae7b6d30ed4554b908c44411cb3e9e74c49e1942ef93c5fe8fad5b5005c23'
SEEDS=(1471,1571,1671);BR=('tap_compare','literal_control')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
    b=p.read_bytes();need(pin is None or sha256(b).hexdigest()==pin,'digest mismatch')
    return json.loads(b)
def save(p,x):
    with p.open('x') as f:json.dump(x,f,allow_nan=False,sort_keys=True,indent=2);f.write('\n')
    print(h(p))
AD=Path('repair_actor_admissions');R=Path('repair_actor_runs')
def roster():
    return {'native-1371-0'}|{f'{arm}-{seed}-{e}' for arm in BR for seed in SEEDS for e in (128,1280)}
def seal(ledger_pin):
    led=read(AD/'ledger.json',ledger_pin); out={};need(set(led)==roster(),'exact case roster')
    for label,c in led.items():
        case=AD/label; r=read(case/'cli-output.json'); run=R/r['id']
        need(verify_run(run)['valid'] and r==read(run/'run.json') and r['status']=='succeeded'
            and r['inputs']==c['inputs'] and r['seed']==c['seed'] and r['name']==label and r['elapsed_seconds']<=30,
            'native actor receipt')
        need(read(case/'recipe.json')==read(run/'experiment.json') and h(case/'recipe.json')==c['recipe_sha256']
            and all(h(case/k)==v for k,v in c['inputs'].items()),'recipe custody')
        need(r['metrics']==dict(prediction_rows=960,model_forwards=120,optimizer_updates=0)
            and set(r['artifacts'])=={'predictions.json'},'actor scope')
        out[label]=dict(id=r['id'],receipt_sha256=h(run/'run.json'),predictions_sha256=r['artifacts']['predictions.json'])
    need(len({x['id'] for x in out.values()})==13,'duplicate actor')
    save(Path('REPAIR_PREDICTION_SEAL.json'),dict(ledger_sha256=ledger_pin,results=out))

def rank(logits,a,b):
    need(len(logits)==287 and all(math.isfinite(x) for x in logits),'logits')
    val=logits[a]
    return all(x<val for i,x in enumerate(logits) if i!=a),val>logits[b]
def score(ledger_pin,seal_pin):
    led=read(AD/'ledger.json',ledger_pin);s=read(Path('REPAIR_PREDICTION_SEAL.json'),seal_pin)
    need(s['ledger_sha256']==ledger_pin and set(s['results'])==set(led)==roster(),'sealed case roster')
    actors=read(Path('repair_panel/actor.json'),A)['prompts']
    judges=read(Path('repair_panel/judge.json'),J)['families']
    need(len(actors)==len(set(actors))==960 and len(judges)==120,'panel')
    worlds=[]
    for i,f in enumerate(judges):
        need(f['rows']==list(range(8*i,8*i+8)),'row geometry')
        family=[]
        for k,z in enumerate(actors[8*i:8*i+8]):
            x,s1,v,y,s2,w,q=z.split();need(s1==s2=='is' and x!=y and v!=w and q in (x,y),'grammar')
            t,u,side=(v,w,'first') if q==x else (w,v,'second')
            need(t==f['answers'][k],'canonical truth mismatch')
            family.append((t,u,side))
        for k in (0,2,4,6):
            need(actors[8*i+k].split()[:6]==actors[8*i+k+1].split()[:6] and
                 family[k][0]==family[k+1][1] and family[k][2]!=family[k+1][2],'query contrast')
        worlds.append((family,'familiar' if f['familiar_name_pair'] else 'unseen_name_pair'))
    out={};base_vocab=None
    for label,c in led.items():
        ss=s['results'][label];d=R/ss['id'];r=read(d/'run.json',ss['receipt_sha256'])
        p=read(d/'output/predictions.json',ss['predictions_sha256'])
        need(r['artifacts']['predictions.json']==ss['predictions_sha256'] and
            (p['arm'],p['seed'],p['updates'],p['backbone_sha256'],p['adapter_sha256'],p['actor_sha256'])==
            (c['arm'],c['seed'],c['endpoint'],c['backbone_sha256'],c['adapter_sha256'],A),'frozen and adapter binding')
        v=p['tokenizer_words'];base_vocab=v if base_vocab is None else base_vocab
        need(v==base_vocab and len(set(v))==25 and p['vocab_size']==287 and len(p['logits'])==960,'vocabulary geometry')
        idx={k:i+262 for i,k in enumerate(v)}
        z={k:Counter() for k in ('all','familiar','unseen_name_pair')}
        for i,(world,group) in enumerate(worlds):
            hits=[rank(l,idx[a],idx[b]) for l,(a,b,_) in zip(p['logits'][8*i:8*i+8],world,strict=True)]
            strict,rel=zip(*hits)
            n=dict(families=1,worlds=8,strict_worlds=sum(strict),strict_families=all(strict),
                   relative_worlds=sum(rel),relative_families=all(rel),
                   first_strict=sum(h and e[2]=='first' for h,e in zip(strict,world)),
                   second_strict=sum(h and e[2]=='second' for h,e in zip(strict,world)),
                   exact_both_queries=sum(strict[n] and strict[n+1] for n in (0,2,4,6)),
                   exact_both_orders=sum(strict[n] and strict[n+2] for n in (0,1,4,5)),
                   exact_both_assignments=sum(strict[n] and strict[n+4] for n in range(4)))
            for g in ('all',group):z[g].update(n)
        out[label]=z
    native=out['native-1371-0']['all'];decisions={}
    for seed in SEEDS:
        a,b=(out[f'{t}-{seed}-1280']['all'] for t in BR)
        cutoff=lambda z:z['strict_worlds']>=912 and z['strict_families']>=108
        decisions[str(seed)]=dict(content_clears=cutoff(a),literal_clears=cutoff(b),
           content_both_query_gain=a['exact_both_queries']-b['exact_both_queries'],
           content_first_minus_native=a['first_strict']-native['first_strict'])
    pref=all(d['content_clears'] and d['content_both_query_gain']>0 and
             d['content_first_minus_native']>=-5 for d in decisions.values())
    save(Path('REPAIR_RESULTS.json'),dict(scores=out,per_seed_primary=decisions,
         content_arm_preferred_on_known_backbone=pref,backbone_robustness_claim=False))
if __name__=='__main__':
    op,lp,*a=sys.argv[1:]
    if op=='seal':seal(lp)
    elif op=='score':score(lp,a[0])
    else:raise ValueError('unsupported op')
