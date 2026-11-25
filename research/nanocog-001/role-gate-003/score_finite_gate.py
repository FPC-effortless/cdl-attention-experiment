"""Sealed native grid replay, followed by weight-free strict scorer."""
from array import array
from collections import Counter
from hashlib import sha256
import json,math
from pathlib import Path
import sys
from finite_gate_score_core import envelope,strict_and_relative
from trainlab.runner import verify_run
SHOP=Path('../cog-role-diag')
A='44f380f0e934450e3c809162578381e670683602a4e52a4487c89fd2197a8116'
J='d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848'
OLD='430460c8260d09537940a1c5efbd034df411baf3f0d801400121e9d67483d766'
B=['native','corner_0','corner_1','corner_2','corner_3','mid_0_1','mid_0_2','mid_0_3','mid_1_2','mid_1_3','mid_2_3','uniform']
def need(ok,msg):
 if not ok: raise ValueError(msg)
def h(p): return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
 raw=p.read_bytes(); need(pin is None or sha256(raw).hexdigest()==pin,f'pin mismatch {p}')
 return json.loads(raw)
def save(p,v):
 with p.open('x') as f:json.dump(v,f,allow_nan=False,sort_keys=True,indent=2);f.write('\n')
 return h(p)

def native_seal(admissions,runs,pin,dest):
 ledger=read(admissions/'ledger.json',pin)
 need(len(ledger)==2 and {s['seed'] for s in ledger.values()}=={1171,1371},'exact roster')
 out={}
 for tag,c in ledger.items():
  source=admissions/tag
  cli=read(source/'cli-output.json'); p=runs/cli['id']
  need(verify_run(p)['valid'] and cli==read(p/'run.json') and cli['status']=='succeeded'
      and cli['seed']==c['seed'] and cli['name']==tag and 0<cli['elapsed_seconds']<=45,'native audit')
  need(cli['inputs']==c['inputs'] and all(h(source/v)==d for v,d in c['inputs'].items())
      and h(source/'recipe.json')==c['recipe_sha256'] and read(source/'recipe.json')==read(p/'experiment.json'),'recipe changed')
  need(set(cli['artifacts'])=={'grid_receipt.json','logits.f32le'},'unexpected outputs')
  needed={'prediction_rows':256,'model_forwards':448,'optimizer_updates':0}
  need(all(cli['metrics'].get(n)==v for n,v in needed.items()),'compute budget')
  out[tag]=dict(id=cli['id'],receipt_sha256=h(p/'run.json'),artifacts=cli['artifacts'])
 need(len({v['id'] for v in out.values()})==2,'duplicate')
 print(save(dest,dict(admission_ledger_sha256=pin,results=out)))

def full_score(admissions,runs,pin,seal,seal_pin,output):
 ledger=read(admissions/'ledger.json',pin)
 ss=read(seal,seal_pin)
 need(set(ss['results'])==set(ledger) and ss['admission_ledger_sha256']==pin,'sealed roster')
 ap=read(SHOP/'probe_panel/actor.json',A)['prompts']
 jp=read(SHOP/'probe_panel/judge.json',J)['families']
 need(len(ap)==256 and len(jp)==32,'panel')
 old=read(SHOP/'PROBE_SEAL.json',OLD)
 out={}
 for tag,c in ledger.items():
  s=ss['results'][tag];p=runs/s['id'];r=read(p/'run.json',s['receipt_sha256'])
  x=r['artifacts'];need(x==s['artifacts'],'artifact binding')
  receipt=read(p/'output/grid_receipt.json',x['grid_receipt.json'])
  need(receipt['schema']=='cog-role-gate-003.grid.v1' and receipt['branch_order']==B
       and receipt['shape']==[12,256,287] and receipt['logits_sha256']==x['logits.f32le'],'logit schema')
  need(receipt['seed']==c['seed'] and receipt['checkpoint_sha256']==c['checkpoint_sha256'] and receipt['actor_sha256']==A,'checkpoint pin')
  raw=(p/'output/logits.f32le').read_bytes()
  need(sys.byteorder=='little' and len(raw)==12*256*287*4 and sha256(raw).hexdigest()==receipt['logits_sha256'],'raw float pin or architecture mismatch')
  v=array('f');v.frombytes(raw)
  oldseed=old[f"probe-{c['seed']}"]
  prev=read(SHOP/'probe_runs'/oldseed['id']/'output/predictions.json',oldseed['predictions_sha256'])
  need(prev['checkpoint_sha256']==receipt['checkpoint_sha256'] and prev['tokenizer_words']==receipt['tokenizer_words']
       and receipt['vocab_size']==287,'vocab lineage')
  lookup={a:i+262 for i,a in enumerate(receipt['tokenizer_words'])};need(len(lookup)==25,'duplicate vocab')
  need(all(v[i*287+j]==w for i,row in enumerate(prev['logits']['native']) for j,w in enumerate(row)),'baseline parity with previous independently verified run')
  truth=[]
  for n,q in enumerate(ap):
   first,i,v1,second,j,v2,asked=q.split()
   need(i==j=='is' and first!=second and v1!=v2 and asked in (first,second),'grammar')
   correct,foil,side=(v1,v2,'first') if asked==first else (v2,v1,'second')
   need(correct==jp[n//8]['answers'][n%8],'truth mismatch')
   truth.append((lookup[correct],lookup[foil],side))
  need(all(ap[i].split()[:6]==ap[i+1].split()[:6] and truth[i][0]==truth[i+1][1] for i in range(0,256,2)),'pair mapping')
  strict,rel={},{}
  for k,b in enumerate(B):
   both=[strict_and_relative(v[(k*256+i)*287:(k*256+i+1)*287],a,f) for i,(a,f,_) in enumerate(truth)]
   strict[b]=[t[0] for t in both];rel[b]=[t[1] for t in both]
  out[tag]={'strict':envelope(strict,[r[2] for r in truth]),'relative':envelope(rel,[r[2] for r in truth]),
            'run_id':s['id'],'native_bit_exact_previous':True}
 print(save(output,out))
if __name__=='__main__':
 op,ad,ru,pin,*args=sys.argv[1:]
 if op=='seal': native_seal(Path(ad),Path(ru),pin,Path(args[0]))
 elif op=='score': full_score(Path(ad),Path(ru),pin,Path(args[0]),args[1],Path(args[2]))
 else:raise ValueError('unsupported operation')
