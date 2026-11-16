"""NEW 017 data. Familiar primitives; fresh composition families, no protected TEST."""
import random,itertools,json,hashlib
from pathlib import Path
import base_data as b
from canonical import convert
SEED=1716112026
original=b.program
NOVEL=['fanout_snapshot','delete_middle','double_detach']
REGRESSION=['cycle_break','recreate']
def program(f,ids,vals):
 a,c,d,e=ids;x,y,z,w=vals;S=lambda i,v:b.op(1,i,v=v);C=lambda i,j:b.op(2,i,j);L=lambda i,j:b.op(3,i,j);D=lambda i:b.op(4,i);Q=lambda i:b.op(5,i)
 if f=='fanout_snapshot':return [S(a,x),C(c,a),L(d,a),L(e,d),S(a,y),Q(e),C(d,e),S(a,z)]
 if f=='delete_middle':return [S(a,x),L(c,a),L(d,c),C(e,d),D(c),Q(d),S(c,y),Q(d),S(a,z)]
 if f=='double_detach':return [S(a,x),L(c,a),L(d,c),C(e,d),S(c,y),S(d,z),D(a),Q(a),S(a,w),L(c,a),D(a),Q(c),S(a,x)]
 return original(f,ids,vals)
b.program=program
from base_data import op,render,state,NAMES,Memory
def make(split,f,form,i,ids,vals,rng,nonce):
 p=program(f,ids,vals)
 # Nuisance values/identities are sampled independently of semantic payloads.
 ignored=[]
 for k in range(1,6):
  d=rng.randrange(4);s=rng.randrange(4)
  while k in (2,3) and (k,d,s) in b.HELD:d=rng.randrange(4);s=rng.randrange(4)
  ignored.append(dict(op(k,d,s,v=rng.randrange(1,5)),ignored=True))
 p=[p[0]]+ignored+p[1:]
 final_start=len(p)
 p += [op(5,j) for j in range(4)]
 m=Memory(4); text=f'Note {nonce}.\n'; events=[]; records=[]; qp=[]; qt=[]; fqp=[]
 for j,o in enumerate(p):
  if j==final_start: boundary=len(text)
  clause=('Ignore: ' if o.get('ignored') else '')+render(o,form)
  assert len(clause)<=48
  pos=len(text)+len(clause)-1
  text+=clause+'\n'
  k=0 if o.get('ignored') else o['op']; ans=-1
  if k:
   result=m.apply(NAMES[k],o['dst'],j=o['src'] if k in (2,3) else None,v=o['value'] if k==1 else None)
   if k==5: ans=0 if result is None else result;qp.append(pos);qt.append(ans)
  if j>=final_start: fqp.append(pos)
  e=dict(pos=pos,op=k,dst=o['dst'] if k else 0,src=o['src'] if k in (2,3) else 0,value=o['value'] if k==1 else 0,answer=ans,state_after=state(m),field_masks={'dst':k!=0,'src':k in (2,3),'value':k==1})
  events.append(e);records.append(dict(o,form=form,pos=pos))
 assert any(qt[-4:]), 'constant unknown could solve session'
 return dict(text=text,events=events,final_state=state(m),query_positions=qp,query_targets=qt,allquery_targets=qt,final_query_positions=fqp,ingest_boundary=boundary,family=f,case_id=f'{split}/{f}/form{form}/{i:03}',nonce=nonce,program=records,identity_permutation=ids,value_permutation=vals)

def build():
 rng=random.Random(SEED);perms=list(itertools.permutations(range(4)));vp=list(itertools.permutations(range(1,5)));data={'train':{},'dev':{}};used=set()
 for split in data:
  for f in b.TRAIN_FAMILIES+(NOVEL+REGRESSION if split=='dev' else []):
   strata=['seen','held'] if f in NOVEL+REGRESSION else ['seen']
   for stratum in strata:
    eligible=[p for p in perms if any((o['op'],o['dst'],o['src']) in b.HELD for o in program(f,p,[1,2,3,4]))==(stratum=='held')]
    if not eligible:raise ValueError((f,stratum))
    for form in (0,1):
     rows=[];count=32 if split=='train' else 16
     for i in range(count):
      while True:
       nonce=''.join(rng.choice('jklmnopq') for _ in range(16))
       if nonce not in used:used.add(nonce);break
      c=make(split,f,form,i,list(rng.choice(eligible)),list(rng.choice(vp)),rng,nonce)
      c['case_id']=f'{split}/{f}/{stratum}/form{form}/{i}';c['identity_symbols']=list('ABCD');c['binding_stratum']=stratum;c['panel']='original';rows.append(c)
     assert len({len(c['text']) for c in rows})==1
     data[split][f'{f}_{stratum}_form{form}']=convert(rows)
 return data
if __name__=='__main__':
 d=build();Path('data.json').write_text(json.dumps(d));print({s:sum(map(len,p.values())) for s,p in d.items()})
