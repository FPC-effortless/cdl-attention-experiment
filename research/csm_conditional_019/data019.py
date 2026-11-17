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
def make(f,form,ids,vals,history_true,switch,cid,pair,binding_pair):
 a,b,c,d=ids;x,y,z,w=vals
 p=[event(1,a,value=x),event(1,b,value=z if history_true else w),event(1,c,value=w if history_true else z),event(1,d,value=x)]
 first,second=b,c
 if f=='linked':p.append(event(3,d,b));first=d
 elif f=='snapshot':p.extend([event(2,d,b),event(1,b,value=x)]);first=d
 elif f=='deleted':
  p=[event(1,a,value=x),event(1,b,value=w),event(1,c,value=w),event(1,d,value=x)]
  live,absent=(b,c) if history_true else (c,b);p.extend([event(1,live,value=z),event(4,absent)])
 elif f=='chain':
  p=[event(1,a,value=w if history_true else z),event(1,b,value=z if history_true else w),event(1,c,value=x),event(1,d,value=x),event(3,c,b),event(3,d,c)];first,second=d,a
 elif f=='detached':p.extend([event(3,d,b),event(1,d,value=z if history_true else w),event(1,b,value=x)]);first=d
 p.append(event(6,a,second if switch else first,value=y,expected=z));p.extend(event(5,i) for i in range(4))
 m=Memory(4);text='Records A B C D.\n';events=[];precondition=None;truth=None
 for e in p:
  clause=render(e,form);assert len(clause)<=48
  pos=len(text)+len(clause)-1;text+=clause+'\n';ans=-1;gate=None
  if e['op']==6:
   precondition=state(m);gate=m.read(e['src'])==e['expected'];truth=gate
   if gate:m.apply('SET',e['dst'],v=e['value'])
  else:ans=m.apply(OPS[e['op']],e['dst'],j=e['src'] if e['op'] in (2,3) else None,v=e['value'] if e['op']==1 else None)
  events.append(dict(e,pos=pos,gate=gate,answer=(0 if ans is None else ans) if e['op']==5 else -1,state_after=state(m)))
 return dict(case_id=cid,pair_id=pair,binding_pair_id=binding_pair,text=text,events=events,final_state=state(m),precondition_state=precondition,family=f,form=form,truth=truth,history_true=history_true,switched_source=switch)
def build():
 data={'train':{},'dev':{}}
 for split in data:
  for f in FAMILIES[:4] if split=='train' else FAMILIES:
   for form in range(2) if split=='train' else (0,4,5):
    rows=[]
    for n in range(16):
     rotation=n//4;ids=[(rotation+j)%4 for j in (range(4) if split=='train' else (0,2,1,3))];vs=[(n+j)%4+1 for j in range(4)];group=f'{split}/{f}/{form}/{n}'
     for switch in (False,True):
      for truth in (False,True):
       history=truth != switch;cid=f'{group}/{int(switch)}/{int(truth)}'
       rows.append(make(f,form,ids,vs,history,switch,cid,group+f'/statepair{int(switch)}',group+f'/bindingpair{int(history)}'))
    maxlen=max(len(c['text']) for c in rows)
    for c in rows:
     pad=maxlen-len(c['text']);c['text']=' '*pad+c['text']
     for e in c['events']:e['pos']+=pad
    data[split][f'{f}_form{form}']=rows
 return data
if __name__=='__main__':
 d=build();Path('data.json').write_text(json.dumps(d));print({s:sum(map(len,p.values())) for s,p in d.items()})
