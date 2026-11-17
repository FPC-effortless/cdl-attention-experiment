"""Paired constructed narrative sessions; not natural-document annotations.
All answer annotations are training/evaluation labels, never inference inputs.
"""
import json,random,hashlib
from pathlib import Path
NAMES={'train':['Mira','Jonas','Sana','Tobin','Leah','Damon','Ravi','Elena'], 'dev':['Nora','Felix','Amina','Cyrus','Inez','Kofi','Luca','Zara']}
STYLE={
 'train':{'set':['The current code for {d} is {v}.','Update {d}: the code is now {v}.','{d} now uses code {v}.'], 'copy':["Save a snapshot of {s}'s code for {d}.","Give {d} a copy of {s}'s current code."], 'link':["Keep {d}'s code linked to {s}.","Let {d} always use {s}'s current code."], 'delete':['Remove the code for {d}.','{d} no longer has a code.'], 'query':['What code does {d} currently have?','Report the present code for {d}.']},
 'dev':{'set':["Replace {d}'s code with {v}.",'For {d}, the newly assigned code is {v}.'], 'copy':["Record {s}'s code as an independent snapshot for {d}.",'Make {d} retain the value {s} has at this moment.'], 'link':["Have {d} follow any changes to {s}'s code.","The code of {d} should mirror {s}'s live code."], 'delete':["Erase {d}'s assigned code.",'Clear the code belonging to {d}.'], 'query':['Which code is assigned to {d} now?',"State {d}'s code as it currently stands."]}}
FILLER=['The meeting continued while the team reviewed the ordinary agenda.','Outside, the weather had changed and several people discussed the journey.','A note about the office schedule was added without changing any codes.']
def read(state,n):
 seen=set()
 while n in state:
  if n in seen:return 'unknown'
  seen.add(n);kind,v=state[n]
  if kind=='value':return v
  n=v
 return 'unknown'
def apply(state,op,d,s=None,v=None):
 if op=='set':state[d]=('value',v)
 elif op=='copy':
  value=read(state,s)
  if value=='unknown':state.pop(d,None)
  else:state[d]=('value',value)
 elif op=='link':state[d]=('ref',s)
 elif op=='delete':state.pop(d,None)
 else:raise ValueError(op)
def episode_pair(split,idx,delay,wording='train',names='train',structure='base'):
 rng=random.Random(f'W020:{split}:{idx}');ns=rng.sample(NAMES[names],4);A,B,C,D=ns
 vals=rng.sample([''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=3)) for _ in range(32)],4)
 while len(set(vals))!=4:vals=[''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=3)) for _ in range(4)]
 # Both branches have exactly same wording/random choices and different earlier fact.
 records=[]
 for variant in [0,1]:
  r=random.Random(f'W020word:{split}:{idx}');state={};parts=[];answers=[];ops=[]
  def pos():return sum(map(len,parts))
  def emit(op,d,s=None,v=None):
   begin=pos();apply(state,op,d,s,v);parts.append(r.choice(STYLE[wording][op]).format(d=d,s=s,v=v)+'\n');ops.append({'op':op,'dst':d,'src':s,'value':v,'start':begin,'end':pos()})
  def query(d,kind):
   qstart=pos();parts.append(r.choice(STYLE[wording]['query']).format(d=d)+'\nAnswer: ');start=pos();val=read(state,d);parts.append(val+'\n');answers.append({'start':start,'end':start+len(val),'question_start':qstart,'target':val,'identity':d,'kind':kind,'after_operations':len(ops)})
  emit('set',A,v=vals[variant]);emit('set',D,v=vals[2]);emit('copy',B,A);emit('link',C,A)
  if structure=='chain':emit('link',C,B)
  elif structure!='base':raise ValueError(structure)
  # Preserve original payload only in snapshot; link follows update for base.
  emit('set',A,v=vals[3]);suffix_start=pos()
  fill=0
  while fill<delay:
   s=r.choice(FILLER)+'\n';parts.append(s);fill+=len(s)
  # First query differs between pair members, suffix immediately preceding it identical.
  query(B,'snapshot');query(C,'live_or_chain');query(D,'untouched')
  emit('delete',A);query(A,'absent');query(C,'post_delete');emit('set',A,v=vals[2]);query(C,'recreation')
  text=''.join(parts)
  records.append({'id':f'{split}-{idx:06d}-v{variant}','group':f'{split}-{idx:06d}','pair_id':f'{split}-{idx:06d}','variant':variant,'text':text,'answers':answers,'operations':ops,'delay_bytes':fill,'requested_delay_bytes':delay,'matched_suffix_start':suffix_start,'structure':structure,'wording':wording,'names':names,'constructed':True})
 return records

def main():
 out=Path(__file__).parent/'private_inputs';out.mkdir(exist_ok=True)
 configs=[('train',4096,[64,256],'train','train','base'),('dev_id',128,[256],'train','train','base'),('dev_names',128,[256],'train','dev','base'),('dev_wording',128,[256],'dev','train','base'),('dev_structure',128,[256],'train','train','chain'),('dev_long',128,[1024,2048],'train','train','base'),('dev_joint',128,[1024,2048],'dev','dev','chain')]
 manifest=[]
 for split,n,delays,w,ns,st in configs:
  path=out/(split+'.jsonl');rows=[v for i in range(n) for v in episode_pair(split,i,delays[i%len(delays)],w,ns,st)]
  path.write_text(''.join(json.dumps(v)+'\n' for v in rows));manifest.append({'split':split,'records':len(rows),'independent_groups':n,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
 (out/'sessions_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':main()
