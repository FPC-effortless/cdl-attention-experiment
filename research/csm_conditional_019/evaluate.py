"""Full closed-loop scoring. No target fields enter forward execution."""
import torch
from model import tensor
import executor
def score_dataset(model,panels):
 result={}
 with torch.inference_mode():
  for name,cases in panels.items():
   logits,answers,final,trace=model.segment(tensor(cases),model.init(len(cases)),trace=True)
   choices=[p.argmax(-1) for p in logits.split([7,4,4,4,4,2],-1)]
   records=[]
   for i,c in enumerate(cases):
    pred=[]
    for t in range(len(c['text'])):
     k,d,s,v,e,g=[int(p[i,t]) for p in choices];v+=1;e+=1
     if k:
      if k==6 and model.arm=='local_exact':
       before=executor.init(len(cases)) if t==0 else (trace[0][:,t-1],trace[1][:,t-1]);g=int(executor.read(before,choices[2][:,t])[i]==e)
      pred.append([t,k,d,s if k in (2,3,6) else 0,v if k in (1,6) else 0,e if k==6 else 0,g if k==6 else -1])
    gold=[[e['pos'],e['op'],e['dst'],e['src'] if e['op'] in (2,3,6) else 0,e['value'] if e['op'] in (1,6) else 0,e['expected'] if e['op']==6 else 0,int(e['gate']) if e['op']==6 else -1] for e in c['events']]
    q=[e for e in c['events'] if e['op']==5]
    states=[{'kind':trace[0][i,e['pos']].tolist(),'data':trace[1][i,e['pos']].tolist()} for e in c['events']]
    records.append(dict(case_id=c['case_id'],pair_id=c['pair_id'],gold_events=gold,pred_events=pred,gold_answers=[e['answer'] for e in q],pred_answers=[int(answers[i,e['pos']]) for e in q],gold_states=[e['state_after'] for e in c['events']],pred_states=states,gold_final=c['final_state'],pred_final={'kind':final[0][0][i].tolist(),'data':final[0][1][i].tolist()}))
   result[name]={'records':records,'metrics':metrics(records),'truth_metrics':{str(t):metrics([r for r in records if next(e for e in r['gold_events'] if e[1]==6)[-1]==t]) for t in (0,1)},'pair_metrics':pair_metrics(records,cases)}
 return result
def metrics(records):
 if not records:raise ValueError('empty panel')
 nq=correct=maps=states=sessions=tp=fp=fn=conddec=0
 for r in records:
  if len(r['gold_answers'])!=4 or len(r['pred_answers'])!=4:raise ValueError('four queries required')
  matches=[a==b for a,b in zip(r['gold_answers'],r['pred_answers'])];correct+=sum(matches);nq+=4;maps+=all(matches);states+=r['gold_final']==r['pred_final']
  sessions+=all(matches) and r['gold_states']==r['pred_states'] and r['gold_events']==r['pred_events']
  gold={e[0]:e for e in r['gold_events'] if e[1]==6};pred={e[0]:e for e in r['pred_events'] if e[1]==6}
  for t,e in gold.items():
   p=pred.get(t);conddec+=p==e
   if e[-1]:
    if p==e:tp+=1
    else:fn+=1
  for t,p in pred.items():
   if p[-1] and p!=gold.get(t):fp+=1
 n=len(records)
 return dict(query_accuracy=correct/nq,complete_map=maps/n,final_state=states/n,complete_session=sessions/n,conditional_decision=conddec/n,conditional_precision=tp/(tp+fp) if tp+fp else 0.,conditional_recall=tp/(tp+fn) if tp+fn else 0.,counts=dict(queries=nq,correct=correct,maps=maps,states=states,sessions=sessions,tp=tp,fp=fp,fn=fn))

def pair_metrics(records,cases):
 from collections import defaultdict
 lookup={r['case_id']:r for r in records};out={}
 for field in ('pair_id','binding_pair_id'):
  groups=defaultdict(list)
  for c in cases:groups[c[field]].append(lookup[c['case_id']])
  if any(len(v)!=2 for v in groups.values()):raise ValueError('paired roster')
  decision=session=0
  for rows in groups.values():
   decision+=all(metrics([r])['conditional_decision']==1 for r in rows)
   session+=all(metrics([r])['complete_session']==1 for r in rows)
  out[field]={'pairs':len(groups),'both_conditional_correct':decision,'both_sessions_correct':session,'conditional_pair_accuracy':decision/len(groups),'session_pair_accuracy':session/len(groups)}
 return out
