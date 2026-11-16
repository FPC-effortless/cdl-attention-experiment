"""Prospective finite typed-operation data; labels derive programs, not parsing."""
import json, random, itertools, hashlib, re, copy
from pathlib import Path
from collections import Counter, defaultdict
from reference_executor import Memory
ROOT=Path(__file__).resolve().parents[1]
SEED=14014026
HELD={(2,0,1),(2,2,3),(3,1,2),(3,3,0)}
NAMES={1:'SET',2:'COPY',3:'LINK',4:'DELETE',5:'QUERY'}
TRAIN_FAMILIES=['overwrite','snapshot','live','delete','link_absent','copy_absent','detach','mixed_ignore']
NOVEL=['held_copy','held_link','branch','recreate','chain_snapshot','cycle_break','self_link','selfcopy_absence_detach']
def op(k,d,s=0,v=0): return dict(op=k,dst=d,src=s if k in (2,3) else 0,value=v if k==1 else 0)
def program(f,ids,vals):
 a,b,c,d=ids; x,y,z,w=vals
 S=lambda i,v:op(1,i,v=v)
 C=lambda i,j:op(2,i,j)
 L=lambda i,j:op(3,i,j)
 D=lambda i:op(4,i)
 Q=lambda i:op(5,i)
 if f=='overwrite': p=[S(a,x),S(b,y),S(c,z),S(d,w),S(a,y)]
 elif f in ('snapshot','held_copy'): p=[S(a,x),C(b,a),S(a,y),S(d,z)]
 elif f in ('live','held_link'): p=[S(a,x),L(b,a),S(a,y),S(d,z)]
 elif f=='delete': p=[S(a,x),S(b,y),S(c,z),S(d,w),D(a)]
 elif f=='link_absent': p=[L(b,a),Q(b),S(a,x),S(d,z)]
 elif f=='copy_absent': p=[C(b,a),S(a,x),S(d,z)]
 elif f=='detach': p=[S(a,x),L(b,a),S(b,y),S(a,z),S(d,w)]
 elif f=='mixed_ignore': p=[S(a,x),S(b,y),L(c,a),D(b)]
 elif f=='branch': p=[S(a,x),C(b,a),L(c,a),S(a,y),S(d,z)]
 elif f=='recreate': p=[S(a,x),L(b,a),D(a),Q(b),S(a,y),Q(b),S(d,z)]
 elif f=='chain_snapshot': p=[S(d,x),L(c,d),L(b,c),L(a,b),Q(a),C(b,a),S(d,y)]
 elif f=='cycle_break': p=[S(d,z),L(a,b),L(b,a),Q(a),Q(b),S(b,x),Q(a)]
 elif f=='self_link': p=[S(a,x),S(d,z),L(a,a),Q(a),S(a,y),Q(a)]
 elif f=='selfcopy_absence_detach': p=[S(a,x),C(a,a),C(b,c),S(c,y),L(b,c),S(b,z),S(c,w),S(d,x)]
 else: raise ValueError(f)
 return p

def render(o,form):
 k=o['op']; d='ABCD'[o['dst']]; s='ABCD'[o['src']]; v='abcd'[max(0,o['value']-1)]
 return {1:(f'Set {d} to {v}.',f'Store {v} in {d}.'),2:(f'Copy {d} from {s}.',f'Snapshot {s} into {d}.'),3:(f'Link {d} to {s}.',f'Alias {d} to {s}.'),4:(f'Delete {d}.',f'Remove {d}.'),5:(f'Read {d}.',f'What is in {d}?')}[k][form]
def state(m): return [[{'absent':0,'value':1,'ref':2}[c.kind],0 if c.kind=='absent' else c.payload] for c in m.cells]
def make_case(split,f,form,i,ids,vals,rng,nonce):
 p=program(f,ids,vals)
 # Nuisance values/identities are sampled independently of semantic payloads.
 ignored=[dict(op(1,rng.randrange(4),v=rng.randrange(1,5)),ignored=True),dict(op(5,rng.randrange(4)),ignored=True)]
 p=[p[0],ignored[0]]+p[1:]
 p=p[:3]+[ignored[1]]+p[3:]
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
 rng=random.Random(SEED); perms=list(itertools.permutations(range(4))); vperms=list(itertools.permutations(range(1,5)))
 old_path=ROOT.parent/'csm-lifecycle-012/data.json'
 old=json.loads(old_path.read_text())
 old_cases=[c for panels in old.values() for rows in panels.values() for c in rows]
 old_nonces={c['nonce'] for c in old_cases}; old_histories={c['text'] for c in old_cases}
 data={'train':{},'dev':{}}; used=set(old_nonces); coverage={}; histories={}
 for split in data:
  families=TRAIN_FAMILIES if split=='train' else TRAIN_FAMILIES+NOVEL
  for f in families:
   for form in range(2):
    cond=f'{f}_form{form}';cases=[]
    eligible=[ids for ids in perms if f not in TRAIN_FAMILIES or not any((o['op'],o['dst'],o['src']) in HELD for o in program(f,ids,[1,2,3,4]))]
    if f=='held_copy': eligible=[ids for ids in perms if (2,ids[1],ids[0]) in HELD]
    if f=='held_link': eligible=[ids for ids in perms if (3,ids[1],ids[0]) in HELD]
    assert eligible
    # Separate shuffled identity/value permutation cycles, not value keyed by identity.
    idseq=[];vseq=[]
    while len(idseq)<64:
     temp=eligible[:];rng.shuffle(temp);idseq+=temp
    while len(vseq)<64:
     temp=vperms[:];rng.shuffle(temp);vseq+=temp
    for i in range(64):
     while True:
      nonce=''.join(rng.choice('jklmnopq') for _ in range(10))
      if nonce not in used:used.add(nonce);break
     case=make_case(split,f,form,i,list(idseq[i]),list(vseq[i]),rng,nonce)
     case['identity_symbols']=['A','B','C','D']
     case['logical_id_to_symbol']={str(j):symbol for j,symbol in enumerate(case['identity_symbols'])}
     case['panel']='original'
     cases.append(case)
    assert len({len(c['text']) for c in cases})==1
    data[split][cond]=cases
  allcases=[c for rows in data[split].values() for c in rows]
  ops=Counter();dst=defaultdict(Counter);src=defaultdict(Counter);values=Counter();answers=Counter();bindings=Counter(); forms=Counter()
  for c in allcases:
   for r,e in zip(c['program'],c['events']):
    if split=='train':assert (r['op'],r['dst'],r['src']) not in HELD
    ops[e['op']]+=1
    if e['op']:
     dst[e['op']][e['dst']]+=1;forms[f"{e['op']}/{r['form']}"]+=1
    if e['op'] in (2,3):src[e['op']][e['src']]+=1;bindings[f"{e['op']}/{e['dst']}/{e['src']}"]+=1
    if e['op']==1:values[e['value']]+=1
    if e['op']==5:answers[e['answer']]+=1
  assert set(ops)==set(range(6)) and set(values)=={1,2,3,4} and set(answers)==set(range(5))
  assert all(set(v)==set(range(4)) for v in dst.values())
  assert all(set(v)==set(range(4)) for v in src.values())
  assert len(forms)==10
  coverage[split]=dict(cases=len(allcases),conditions=len(data[split]),bytes=sum(len(c['text']) for c in allcases),events=dict(ops),dst_by_op=dict(dst),src_by_op=dict(src),set_values=dict(values),query_answers=dict(answers),bindings=dict(bindings),forms=dict(forms),condition_lengths={k:len(v[0]['text']) for k,v in data[split].items()},query_count=sum(len(c['query_targets']) for c in allcases))
  histories[split]={c['text'] for c in allcases}
 assert not histories['train']&histories['dev']
 assert all(coverage['dev']['bindings'].get('/'.join(map(str,b)),0)>0 for b in HELD)
 assert not (histories['train']|histories['dev']) & old_histories
 assert not {c['nonce'] for split in data.values() for rows in split.values() for c in rows} & old_nonces
 # Paired renamed DEV reuses the exact logical program and nonce within DEV only.
 # Single uppercase word tokens name identities; command-initial letters are untouched.
 rename_map=dict(zip('ABCD','EFGH')); identity_token=re.compile(r'\b[A-D]\b')
 original_dev=list(data['dev'].items())
 for cond, rows in original_dev:
  renamed=[]
  for c in rows:
   r=copy.deepcopy(c); r['text']=identity_token.sub(lambda m:rename_map[m.group()],c['text'])
   r['paired_original_case_id']=c['case_id']; r['case_id']=c['case_id']+'/renamed_EH'
   r['identity_symbols']=['E','F','G','H'];r['logical_id_to_symbol']={str(j):v for j,v in enumerate(r['identity_symbols'])};r['panel']='renamed'
   assert len(r['text'])==len(c['text'])
   assert all(a==b or (a in rename_map and rename_map[a]==b) for a,b in zip(c['text'],r['text']))
   assert not identity_token.search(r['text'])
   for key in ('events','program','query_positions','query_targets','final_state','final_query_positions','ingest_boundary'):
    assert r[key]==c[key]
   renamed.append(r)
  data['dev'][cond+'_renamed_EH']=renamed
 assert not any(re.search(r'\b[E-H]\b',c['text']) for rows in data['train'].values() for c in rows)
 coverage['dev_original']=coverage.pop('dev')
 coverage['dev_renamed']=copy.deepcopy(coverage['dev_original'])
 coverage['dev_renamed']['condition_lengths']={k+'_renamed_EH':v for k,v in coverage['dev_original']['condition_lengths'].items()}
 counts=dict(seed=SEED,held_bindings=sorted(HELD),splits=coverage,unique_nonces=len(used-old_nonces),paired_dev_programs=2048,dev_total_cases=4096,original_012_history_overlap=0,original_012_nonce_overlap=0,train_EH_identity_tokens=0,original_012_data_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),original_012_generator_sha256=hashlib.sha256((ROOT/'source/original_data_build_012.py').read_bytes()).hexdigest(),history_intersection=0,protected_test_access=False,source_reference_sha256=hashlib.sha256((ROOT/'source/reference_executor.py').read_bytes()).hexdigest(),scope='finite typed operations; paired E-H identity renaming shares DEV programs and is not independent structural evidence; no inference role resolver supplied')
 (ROOT/'data.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
 (ROOT/'DATA_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
 (ROOT/'DATA_COVERAGE.md').write_text('# Prospective pointer-014 data coverage\n\nGenerated deterministically before model execution with seed '+str(SEED)+'.\n\n'+ '\n'.join(f'- {s}: {c["cases"]} histories, {c["conditions"]} uniform-length conditions, {c["query_count"]} actual learned-query targets.' for s,c in coverage.items())+'\n\nAll six event classes, both render forms for every real operation, all four destinations per operation, all four sources for COPY/LINK, all four SET payloads, and all five query answer classes occur in TRAIN and DEV. Held bindings are absent from every TRAIN operation record including ignored commands and present in DEV. Nonces and full histories are disjoint. Every history has at least one non-absent final query. Final four queries are executed in a single persistent session. Extra intermediate queries test absence, chain, and cycle transitions.\n\nIgnored SET and QUERY clauses have independently drawn nuisance identities/payloads and are interspersed. No regex/parser produces gold labels: explicit operation records are rendered, then the frozen pure reference executes those records. States are audit labels only, never neural inputs.\n\nEach family has two fixed-form conditions, 64 cases each; DEV therefore has 128 cases per family across two forms. Identity and value permutations are independently shuffled prospective schedules, not an exact factorial design. Final query ordering is A,B,C,D in original panels and E,F,G,H in paired renamed panels; correct selection must still be learned. Original A-D panels use familiar identities/values/forms. Paired E-H DEV changes only single-uppercase identity tokens; those tokens never occur as TRAIN identities. Each renamed case retains original nonce, positions, logical programs and gold states. The 2,048 paired cases are NOT independent new structures or evidence of broad language generalization. No role resolution is supplied to inference. Fresh original histories and nonces are disjoint from 012 as well as across TRAIN/DEV. There is no protected final TEST.\n')
 print(json.dumps({s:{k:v[k] for k in ('cases','conditions','bytes','query_count')} for s,v in coverage.items()}))
if __name__=='__main__':build()
