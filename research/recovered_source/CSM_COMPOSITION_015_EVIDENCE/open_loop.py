"""Open-loop symbolic diagnostics; corrected actions do not rerun neural policy."""
import json,os,hashlib
from pathlib import Path
from collections import Counter
from reference_executor import Memory
from evaluate import validate_record
MODES=['original','source_onlyCOPYLINK','destination_onlyCOPYLINK','both_identitiesCOPYLINK','full_copy_link','set_payload_only','full_set','remove_false_actions','all_gold_ceiling']
def state(m):return [[{'absent':0,'value':1,'ref':2}[c.kind],0 if c.kind=='absent' else c.payload] for c in m.cells]
def replay(r,seq):
 m=Memory(4);actions={e[0]:e for e in seq};queries={};terminals={};tp=set(r['terminal_positions']);answers={};states=[]
 for p in range(r['length']):
  if p in actions:
   _,op,dst,src,val=actions[p];answer=m.apply(['NOOP','SET','COPY','LINK','DELETE','QUERY'][op],dst,src,val)
   if op==5:queries[p]=0 if answer is None else answer
  if p in tp:terminals[p]=state(m)
 return {'query_predictions':[queries.get(p,-1) for p in r['query_positions']],'final_query_predictions':[queries.get(p,-1) for p in r['final_query_positions']],'pred_terminal_states':[terminals[p] for p in r['terminal_positions']],'pred_final_state':state(m),'pred_emitted_query_answers':[[p,v] for p,v in sorted(queries.items())]}
def patched(r,mode):
 pred={e[0]:e.copy() for e in r['pred_nonnoop']};gold={e[0]:e for e in r['gold_nonnoop']}
 if mode=='all_gold_ceiling':return r['gold_nonnoop']
 if mode=='remove_false_actions':return [e for p,e in pred.items() if p in gold]
 for p,g in gold.items():
  a=pred.get(p)
  if mode=='full_copy_link' and g[1] in (2,3):pred[p]=g.copy()
  elif mode=='full_set' and g[1]==1:pred[p]=g.copy()
  elif a and a[1]==g[1]:
   if g[1] in (2,3):
    if mode in ('destination_onlyCOPYLINK','both_identitiesCOPYLINK'):a[2]=g[2]
    if mode in ('source_onlyCOPYLINK','both_identitiesCOPYLINK'):a[3]=g[3]
   if mode=='set_payload_only' and g[1]==1:a[4]=g[4]
 return [e for p,e in sorted(pred.items())]
def errors(r):
 pred={e[0]:e for e in r['pred_nonnoop']};gold={e[0]:e for e in r['gold_nonnoop']};counts=Counter();peropcode={};first=None
 for p in sorted(set(pred)|set(gold)):
  a=pred.get(p);g=gold.get(p);cats=[]
  if a==g:continue
  if g is None:cats=['false_action']
  elif a is None:cats=['missing_action']
  elif a[1]!=g[1]:cats=['wrong_opcode']
  else:
   # Relevant fields have comparable semantics only with a correct opcode.
   # Source and destination errors can overlap within COPY/LINK.
   if a[2]!=g[2]:cats.append('wrong_destination')
   if g[1] in (2,3) and a[3]!=g[3]:cats.append('wrong_source')
   if g[1]==1 and a[4]!=g[4]:cats.append('wrong_payload')
  key='gold_'+str(g[1]) if g else 'false_pred_'+str(a[1])
  bucket=peropcode.setdefault(key,Counter());bucket['divergent_positions']+=1
  for c in cats:counts[c]+=1;bucket[c]+=1
  if first is None:first={'position':p,'gold':g,'pred':a,'categories':cats,'opcode_group':key}
 return dict(counts),first,{k:dict(v) for k,v in peropcode.items()}

def evaluate(rows):
 output=[]
 for item in rows:
  r=item['record'];validate_record(r);base=replay(r,r['pred_nonnoop'])
  for k,v in base.items():assert v==r[k],(item['arm'],item['seed'],r['case_id'],k)
  ec,first,byopcode=errors(r);modes={};gold=r['query_targets'];basecorrect=[p==g for p,g in zip(base['query_predictions'],gold)]
  for mode in MODES:
   rr=replay(r,patched(r,mode));correct=[p==g for p,g in zip(rr['query_predictions'],gold)];final=[p==g for p,g in zip(rr['final_query_predictions'],r['final_query_targets'])]
   modes[mode]={'queries_correct':sum(correct),'query_count':len(gold),'complete_map':all(final),'final_state_correct':rr['pred_final_state']==r['gold_final_state'],'terminal_states_correct':rr['pred_terminal_states']==r['gold_terminal_states'],'rescued_queries':sum(c and not b for c,b in zip(correct,basecorrect)),'damaged_queries':sum(b and not c for c,b in zip(correct,basecorrect)),'query_predictions':rr['query_predictions']}
  assert modes['all_gold_ceiling']['complete_map'] and modes['all_gold_ceiling']['queries_correct']==len(gold) and modes['all_gold_ceiling']['final_state_correct'] and modes['all_gold_ceiling']['terminal_states_correct']
  output.append({'arm':item['arm'],'seed':item['seed'],'panel':item['panel'],'case_id':r['case_id'],'errors':ec,'errors_by_opcode':byopcode,'field_counts_overlap':True,'first_divergence':first,'modes':modes})
 return output
