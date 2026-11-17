"""Constructed conditional histories; draft generator, never protected TEST."""
import itertools,json,random
from pathlib import Path
from reference_executor import Memory
OPS=['NOOP','SET','COPY','LINK','DELETE','QUERY','IFSET']
FAMILIES=['direct','linked','snapshot','deleted','chain','detached']
def event(op,dst=0,src=0,value=1,expected=1):return dict(op=op,dst=dst,src=src,value=value,expected=expected)
def render(e,form):
 k,i,j,v,c=e['op'],chr(65+e['dst']),chr(65+e['src']),chr(96+e['value']),chr(96+e['expected'])
 patterns={
 1:[f'Set {i} to {v}.',f'Put {v} in {i}.',f'Assign {v} to {i}.',f'Write {v} into {i}.',f'Replace {i} with {v}.',f'Give {i} the value {v}.'],
 2:[f'Copy {j} into {i}.',f'Snapshot {j} to {i}.',f'Put a copy of {j} in {i}.',f'Copy from {j} to {i}.',f'Let {i} keep a copy of {j}.',f'Duplicate {j} into {i}.'],
 3:[f'Link {i} to {j}.',f'Let {i} follow {j}.',f'Make {i} refer to {j}.',f'Alias {i} to {j}.',f'Have {i} track {j}.',f'Keep {i} linked to {j}.'],
 4:[f'Delete {i}.',f'Clear {i}.',f'Remove {i}.',f'Erase {i}.',f'Make {i} absent.',f'Discard {i}.'],
 5:[f'Query {i}?',f'Read {i}?',f'Value of {i}?',f'What is {i}?',f'What does {i} hold?',f'Give the value of {i}?'],
 6:[f'If {j} is {c}, set {i} to {v}.',f'Set {i} to {v} if {j} is {c}.',f'When {j} holds {c}, write {v} to {i}.',f'Write {v} to {i} only if {j} is {c}.',f'Provided {j} is {c}, put {v} in {i}.',f'Give {i} value {v} when {j} holds {c}.']}
 return patterns[k][form]
def state(m):return {'kind':[{'absent':0,'value':1,'ref':2}[c.kind] for c in m.cells],'data':[0 if c.payload is None else c.payload for c in m.cells]}
def make(f,form,ids,vals,truth,cid,pair):
 a,b,c,d=ids;x,y,z,w=vals
 # Destination starts at x; proposed y differs; source z vs w is balanced.
 p=[event(1,a,value=x),event(1,b,value=z if truth else w),event(1,c,value=x),event(1,d,value=w)]
 src=b
 if f=='linked':p.append(event(3,c,b));src=c
 elif f=='snapshot':p+=[event(2,c,b),event(1,b,value=x)];src=c
 elif f=='deleted':
  # balanced deleted/present histories rather than constant false deletion family
  if not truth:p.append(event(4,b))
 elif f=='chain':p+=[event(3,c,b),event(3,d,c)];src=d
 elif f=='detached':p+=[event(3,c,b),event(1,c,value=z if truth else w),event(1,b,value=x)];src=c
 cond=event(6,a,src,value=y,expected=z);p.append(cond)
 p.extend(event(5,i) for i in range(4))
 m=Memory(4);text='Records A B C D.\n';events=[]
 for e in p:
  clause=render(e,form);assert len(clause)<=48
  pos=len(text)+len(clause)-1;text+=clause+'\n';ans=-1;gate=None
  if e['op']==6:
   gate=m.read(e['src'])==e['expected']
   if gate:m.apply('SET',e['dst'],v=e['value'])
  else:ans=m.apply(OPS[e['op']],e['dst'],j=e['src'] if e['op'] in (2,3) else None,v=e['value'] if e['op']==1 else None)
  events.append(dict(e,pos=pos,gate=gate,answer=(0 if ans is None else ans) if e['op']==5 else -1,state_after=state(m)))
 return dict(case_id=cid,pair_id=pair,text=text,events=events,final_state=state(m),family=f,form=form,truth=truth)
def build():
 rng=random.Random(1817112026);perms=list(itertools.permutations(range(4)));vals=list(itertools.permutations(range(1,5)));data={'train':{},'dev':{}}
 # Balanced cyclic value/role roster; DEV swaps source and companion roles.
 # Not all possible value permutations; no unrestricted payload-generalization claim.
 for split in data:
  for f in FAMILIES[:4] if split=='train' else FAMILIES:
   for form in range(4) if split=='train' else (0,4,5):
    rows=[]
    for n in range(16):
     rotation=n//4;ids=[(rotation+j)%4 for j in (range(4) if split=='train' else (0,2,1,3))];vs=[(n+j)%4+1 for j in range(4)];pair=f'{split}/{f}/{form}/{n}'
     rows.extend(make(f,form,ids,vs,t,f'{pair}/{int(t)}',pair) for t in (False,True))
    maxlen=max(len(c['text']) for c in rows)
    for c in rows:
     pad=maxlen-len(c['text']);c['text']=' '*pad+c['text']
     for e in c['events']:e['pos']+=pad
    data[split][f'{f}_form{form}']=rows
 return data
if __name__=='__main__':
 d=build();Path('data.json').write_text(json.dumps(d));print({s:sum(map(len,p.values())) for s,p in d.items()})
