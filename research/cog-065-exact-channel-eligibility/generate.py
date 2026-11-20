"""Freeze constructed observations and separate primitive/evaluator references; scores nothing."""
import hashlib,json,random
from pathlib import Path
from forge_data.core import canonical
from forge_data.diagnostics import diagnose
from primitive_reference import World,initial,step,done
from rules import ROLES,KNOWN
from codec import ORDER,CLAUSES
from timeline import Envelope,snapshot
BASE=Path(__file__).resolve().parent
def write(p,d):
 b=canonical(d)
 if p.exists() and p.read_bytes()!=b:raise ValueError(f'existing frozen bytes differ: {p}')
 p.write_bytes(b)
def pinned(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def payload(k,v,version,tick):
 if version==1:return dict(version=1,subject='parcel request',emitter=k,event=v,available=tick)
 return dict(version=2,scope='parcel request',provenance=k,observation={'kind':v,'tick':tick})
def decode(slot,obj,tick):
 from codec import normalize_receipt
 r=normalize_receipt(obj)
 if r['scope']!='parcel request' or r['issuer']!=slot or r['available']!=tick or r['kind'] not in CLAUSES[slot]:
  raise ValueError('occupied receipt provenance or declared slot inconsistent')
 return r['kind']
def freeze():
 p=BASE/'frozen';p.mkdir(exist_ok=True)
 train=[]; private={}
 for r,role in enumerate(ROLES):
  for i in range(24):
   group=f'{r}-{i}';rng=random.Random(650650+r*100+i);target=rng.randrange(1,4);tick=rng.randrange(5,50);version=1+i%2
   w=World(((0,target),),(2,),('parcel request',));s=initial(w);path=[4,target,4][:3 if i%3==0 else 1+(i%2)]
   for a in path:s,_=step(w,s,a)
   f={'status':'completed' if done(w,s) else 'blocked' if i%3==1 else 'unknown','proof':'accepted' if done(w,s) else 'absent','authority':'current','source':'verified','scope':'matched','freshness':'current','progress':'started'}
   mutation=('scope','authority','source','freshness')[i%4]
   counter={'scope':'mismatch','authority':'revoked','source':'unverified','freshness':'stale'}
   for side in range(2):
    row=f'{group}/{side}';fx=dict(f)
    if side:fx[mutation]=counter[mutation]
    raw=[]
    for k in ORDER:
     # Same-tick superseded malformed bytes and a later unavailable payload
     # exercise the boundary; decoder must not touch either.
     raw.extend([dict(slot=k,tick=tick,sequence=1,payload={'not':'receipt'}),
                 dict(slot=k,tick=tick,sequence=2,payload=payload(k,fx[k],version,tick)),
                 dict(slot=k,tick=tick+1,sequence=0,payload={'invalid':'future'})])
    rng.shuffle(raw)
    got=snapshot([Envelope(**e) for e in raw],tick,decode)
    assert got==fx
    train.append({'id':row,'split':'validation','features':{'role':role,'cutoff':tick,'raw':raw},'groups':{'causal_group':group}})
    private[row]={'group':group,'role':role,'facts':fx,'contexts':[{k:fx[k] for k in ls} for ls in KNOWN[role]],'primitive':{'target':target,'path':path,'complete':done(w,s)},'intervention':mutation,'side':side}
 view={'schema':'forge.diagnostic-input.v1','feature_fields':['role','cutoff','raw'],'group_fields':['causal_group'],'rows':train}
 a=p/'observations.json';write(a,view);write(p/'forge-receipt.json',diagnose(a,sha256=pinned(a),max_rows=1000,max_findings=0));write(p/'private-reference.json',private)
 print('Frozen 72 constructed paired input groups and separate private reference; no comparison')
if __name__=='__main__':freeze()
