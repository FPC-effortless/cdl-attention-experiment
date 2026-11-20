"""Text-only learned parsing, typed execution, raw receipts and independent scoring."""
import hashlib,json,re,time
from pathlib import Path
import torch
from models import OPS,REQUIRED,infer_chunks,parity_guard
from executor import Executor,reference
from data import external_events,payload_int,payload_text,validate_roster
NAME=re.compile(r'[A-Za-z][A-Za-z0-9_]*/[A-Za-z][A-Za-z0-9_]*\Z')

def commands(texts,actions,capacity=64):
 if len(texts)!=len(actions):raise ValueError('action roster mismatch')
 ids={};banks={};rows=[];errors=[]
 for text,a in zip(texts,actions):
  try:
   if not a['valid']:raise ValueError('invalid span')
   op=a['op'];fields={}
   for f in REQUIRED[op]:
    l,r=a['spans'][f]
    if not 0<=l<r<=len(text):raise ValueError('span bounds')
    fields[f]=text[l:r]
    if f in ('dst','src') and not NAME.fullmatch(fields[f]):raise ValueError('invalid identity literal')
    if f in ('value','expected'):payload_int(fields[f])
   # Validate complete action before any registry mutation. No target/role oracle.
   new=sorted(set(fields[f] for f in ('dst','src') if f in fields and fields[f] not in ids),key=text.index)
   if len(ids)+len(new)>capacity:raise ValueError('capacity exceeded')
   for name in new:
    ids[name]=len(ids);bank=name.split('/')[0]
    if bank not in banks:banks[bank]=len(banks)
   rows.append([OPS.index(op),ids[fields['dst']] if 'dst' in fields else 0,ids[fields['src']] if 'src' in fields else 0,payload_int(fields['value']) if 'value' in fields else 0,payload_int(fields['expected']) if 'expected' in fields else 0]);errors.append(None)
  except (ValueError,KeyError,IndexError,TypeError) as e:
   rows.append([0,0,0,0,0]);errors.append(str(e))
 n=max(1,len(ids));bank_ids=[0]*n
 for name,i in ids.items():bank_ids[i]=banks[name.split('/')[0]]
 return rows,n,bank_ids,errors,ids

@torch.no_grad()
def session(model,record,executor,chunk=1,parallel=False,threads=1,capacity=64):
 texts=external_events(record);device=next(model.parameters()).device
 if device.type=='cuda':torch.cuda.synchronize()
 start=time.perf_counter();logits,actions=infer_chunks(model,texts,chunk)
 if device.type=='cuda':torch.cuda.synchronize()
 parser_end=time.perf_counter();ops,n,banks,errors,ids=commands(texts,actions,capacity);registry_end=time.perf_counter()
 result=executor.execute(ops,n,bank_ids=banks,threads=threads,parallel=parallel)
 rendered=[None if a==-2 else ('No code is currently assigned.' if a is None else 'The current code is '+payload_text(a)+'.') for a in result['answers']]
 end=time.perf_counter()
 # Independent serial audit is outside measured production-session cost.
 state=None;guards=[]
 for op in ops:
  guard=None
  if op[0]==6:
   guard=reference([[5,op[2],0,0,0]],n,state)['answers'][0]==op[4]
  step=reference([op],n,state);state=step['state'];guards.append(guard)
 if state!=result['state']:raise RuntimeError('native/reference final-state mismatch')
 row={'id':record['id'],'pair_id':record['pair_id'],'variant':record['variant'],'actions':actions,'typed_commands':ops,'registry':ids,'errors':errors,'answers':result['answers'],'rendered_answers':rendered,'predicted_guards':guards,'final_state':[list(x) for x in result['state']],'executor_stats':result['stats'],'cost':{'parser_including_tokenizer_and_decode_seconds':parser_end-start,'registry_validation_seconds':registry_end-parser_end,'executor_and_render_wall_seconds':end-registry_end,'complete_session_seconds':end-start},'schedule':{'chunk':chunk,'parallel_executor':parallel,'threads':threads}}
 return row,logits

def audit_record(record,row):
 ops,n,banks,errors,registry=commands(external_events(record),row['actions'])
 if row['typed_commands']!=ops or row['errors']!=errors or row['registry']!=registry:raise ValueError('raw-action command/registry/error audit')
 state=None;answers=[];guards=[]
 for op in ops:
  guard=reference([[5,op[2],0,0,0]],n,state)['answers'][0]==op[4] if op[0]==6 else None
  step=reference([op],n,state);answers.extend(step['answers']);state=step['state'];guards.append(guard)
 rendered=[None if a==-2 else ('No code is currently assigned.' if a is None else 'The current code is '+payload_text(a)+'.') for a in answers]
 expected={'answers':answers,'predicted_guards':guards,'rendered_answers':rendered,'final_state':[list(x) for x in state]}
 for name,value in expected.items():
  if row.get(name)!=value:raise ValueError('serial raw-action audit mismatch '+name)
 return expected

def score(records,rows):
 validate_roster(records)
 if len(rows)!=len(records) or len({r['id'] for r in rows})!=len(rows):raise ValueError('prediction roster')
 byid={r['id']:r for r in rows}
 if set(byid)!={r['id'] for r in records}:raise ValueError('prediction ID substitution')
 counts={'queries':0,'answers_correct':0,'nonabsence_queries':0,'nonabsence_correct':0,'absence_queries':0,'absence_correct':0,'complete_sessions':0,'sessions':len(records),'operation_correct':0,'events':0,'all_required_roles_correct':0,'invalid_actions':0,'paired_first_correct':0,'pairs':len(records)//2,'conditional_decisions':0,'conditional_decisions_correct':0}
 pair={}
 for record in records:
  r=byid[record['id']]
  audit_record(record,r)
  if (r['pair_id'],r['variant'])!=(record['pair_id'],record['variant']):raise ValueError('pair/variant substitution')
  if not(len(r['answers'])==len(r['actions'])==len(r['errors'])==len(record['events'])):raise ValueError('event roster')
  complete=True;first=None
  for event_index,(e,a,answer,error) in enumerate(zip(record['events'],r['actions'],r['answers'],r['errors'])):
   g=e['gold'];truth=g['op_name'];counts['events']+=1;counts['invalid_actions']+=error is not None
   counts['operation_correct']+=a['op']==truth
   roles=a['op']==truth and a['valid'] and all(a['spans'][f]==e['labels']['spans'][f] for f in REQUIRED[truth]);counts['all_required_roles_correct']+=roles
   if truth=='IFSET':
    counts['conditional_decisions']+=1
    counts['conditional_decisions_correct']+=error is None and a['op']=='IFSET' and r.get('predicted_guards',[None]*len(record['events']))[event_index]==g['guard']
   if truth=='QUERY':
    counts['queries']+=1;expected=None if g['answer'] is None else payload_int(g['answer'])
    correct=error is None and a['op']=='QUERY' and answer==expected
    counts['answers_correct']+=correct;key='absence' if expected is None else 'nonabsence';counts[key+'_queries']+=1;counts[key+'_correct']+=correct
    complete=complete and correct
    if first is None:first=correct
   elif a['op']=='QUERY':complete=False # unsolicited output is not a clean session
  counts['complete_sessions']+=complete;pair.setdefault(record['pair_id'],[]).append(first)
 counts['paired_first_correct']=sum(len(x)==2 and all(x) for x in pair.values())
 counts['qualified']=counts['complete_sessions']/max(1,counts['sessions'])>=.95 and counts['paired_first_correct']/max(1,counts['pairs'])>=.95
 return counts

def run(model,records,out,chunk=1,parallel=False,threads=1,reference_rows=None,executor=None):
 validate_roster(records);ex=executor or Executor();rows=[];parity=[]
 model.eval();start=time.perf_counter()
 for record in records:
  row,z=session(model,record,ex,chunk,parallel,threads)
  if reference_rows is not None:
   old=reference_rows[record['id']]
   for field in ('actions','typed_commands','registry','errors','answers','rendered_answers','predicted_guards','final_state'):
    if row[field]!=old[field]:raise RuntimeError('end-to-end scheduling parity failed '+record['id']+' '+field)
   parity.append(record['id'])
  rows.append(row)
 result={'rows':rows,'scores':score(records,rows),'end_to_end_parity_sessions':len(parity),'total_seconds_including_scoring':time.perf_counter()-start,'complete_session_seconds':sum(r['cost']['complete_session_seconds'] for r in rows),'scope':'constructed framed instructions; learned spans plus engineered exact executor'}
 Path(out).write_text(json.dumps(result,indent=2));return result
