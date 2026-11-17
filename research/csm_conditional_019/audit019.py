"""Independent saved-record scorer and pure-Python execution consistency audit.
No evaluate/model/executor imports; does not establish checkpoint execution.
"""
import json, math, hashlib
from pathlib import Path

def need(ok, message):
 if not ok: raise ValueError(message)
def state_ok(s):
 need(isinstance(s,dict) and set(s)=={'kind','data'},'state schema')
 need(all(isinstance(s[k],list) and len(s[k])==4 for k in s),'state shape')
 for k,d in zip(s['kind'],s['data']):
  need(type(k) is int and type(d) is int,'state dtype')
  need((k==0 and d==0) or (k==1 and 1<=d<=4) or (k==2 and 0<=d<4),'state value')
def empty():return {'kind':[0]*4,'data':[0]*4}
def read(s,i):
 seen=set()
 while i not in seen:
  seen.add(i);k,d=s['kind'][i],s['data'][i]
  if k==0:return 0
  if k==1:return d
  i=d
 return 0
def event_ok(e,length):
 need(isinstance(e,list) and len(e)==7 and all(type(x) is int for x in e),'event shape/type')
 t,k,d,s,v,c,g=e
 need(0<=t<length and 1<=k<=6 and 0<=d<4,'event bounds')
 need((0<=s<4) if k in (2,3,6) else s==0,'source')
 need((1<=v<=4) if k in (1,6) else v==0,'payload')
 need((1<=c<=4 and g in (0,1)) if k==6 else (c==0 and g==-1),'predicate')
def transition(st,e,exact=False):
 _,k,d,s,v,c,g=e;out={q:list(a) for q,a in st.items()};answer=-1
 if k==6:
  if exact:need(g==int(read(st,s)==c),'exact comparator mismatch')
  if not g:return out,answer
  k=1
 if k==1:out['kind'][d]=1;out['data'][d]=v
 elif k==2:
  value=read(st,s);out['kind'][d]=int(value!=0);out['data'][d]=value
 elif k==3:out['kind'][d]=2;out['data'][d]=s
 elif k==4:out['kind'][d]=out['data'][d]=0
 elif k==5:answer=read(st,d)
 return out,answer

def gold_event(e):
 k=e['op']
 return [e['pos'],k,e['dst'],e['src'] if k in (2,3,6) else 0,e['value'] if k in (1,6) else 0,e['expected'] if k==6 else 0,int(e['gate']) if k==6 else -1]

def validate_record(r,c,arm):
 required={'case_id','pair_id','gold_events','pred_events','gold_answers','pred_answers','gold_states','pred_states','gold_final','pred_final'}
 need(set(r)==required,'record schema')
 need(r['case_id']==c['case_id'] and r['pair_id']==c['pair_id'],'record identity')
 ge=[gold_event(e) for e in c['events']]
 need(r['gold_events']==ge and len([e for e in ge if e[1]==6])==1,'gold events')
 need(r['gold_states']==[e['state_after'] for e in c['events']] and r['gold_final']==c['final_state'],'gold states')
 need(r['gold_answers']==[e['answer'] for e in c['events'] if e['op']==5],'gold answers')
 need(len(r['gold_answers'])==len(r['pred_answers'])==4,'answer count')
 need(all(type(a) is int and -1<=a<=4 for a in r['pred_answers']),'answer dtype/range')
 for key in ('gold_states','pred_states'):
  need(isinstance(r[key],list) and len(r[key])==len(ge),'state checkpoint count')
  for st in r[key]:state_ok(st)
 state_ok(r['gold_final']);state_ok(r['pred_final'])
 for key in ('gold_events','pred_events'):
  need(isinstance(r[key],list),'events list')
  for e in r[key]:event_ok(e,len(c['text']))
  positions=[e[0] for e in r[key]];need(positions==sorted(set(positions)),'event order/duplicates')
 # Verify generated gold semantics independently, not just copied metadata.
 st=empty();ga=[];gs=[]
 for e in ge:
  st,a=transition(st,e,exact=True);gs.append(st)
  if e[1]==5:ga.append(a)
 need(gs==r['gold_states'] and ga==r['gold_answers'] and st==r['gold_final'],'gold semantics')
 # Reconstruct predictions at gold checkpoints, including intervening false writes.
 pred={e[0]:e for e in r['pred_events']};positions={e[0]:e for e in ge};st=empty();pa=[];ps=[]
 for t in sorted(set(pred)|set(positions)):
  a=-1
  if t in pred:st,a=transition(st,pred[t],exact=arm=='local_exact')
  if t in positions:
   ps.append(st)
   if positions[t][1]==5:pa.append(a)
 need(ps==r['pred_states'] and pa==r['pred_answers'] and st==r['pred_final'],'prediction execution consistency')

def recompute(records):
 need(bool(records),'empty scoring group')
 n=len(records);query=maps=final=sessions=decisions=tp=fp=fn=0
 for r in records:
  match=sum(x==y for x,y in zip(r['gold_answers'],r['pred_answers']));query+=match;maps+=match==4;final+=r['gold_final']==r['pred_final']
  sessions+=match==4 and r['gold_states']==r['pred_states'] and r['gold_events']==r['pred_events']
  g=next(e for e in r['gold_events'] if e[1]==6)
  pe=[e for e in r['pred_events'] if e[1]==6]
  exact=g in pe;decisions+=exact
  if g[-1]:tp+=exact;fn+=not exact
  fp+=sum(e[-1]==1 and e!=g for e in pe)
 return {'query_accuracy':query/(4*n),'complete_map':maps/n,'final_state':final/n,'complete_session':sessions/n,'conditional_decision':decisions/n,'conditional_precision':tp/(tp+fp) if tp+fp else 0.,'conditional_recall':tp/(tp+fn) if tp+fn else 0.,'counts':dict(queries=4*n,correct=query,maps=maps,states=final,sessions=sessions,tp=tp,fp=fp,fn=fn)}
def compare(a,b):
 need(type(a) is dict and type(b) is dict and set(a)==set(b),'metric keys')
 for k,v in a.items():
  if isinstance(v,dict):compare(v,b[k])
  else:need(type(b[k]) in (int,float) and math.isfinite(b[k]) and abs(v-b[k])<=1e-12,'metric mismatch '+k)
def validate_quartets(cases):
 """Independent information checks, not generator metadata acceptance."""
 from collections import defaultdict,Counter
 pairs={key:defaultdict(list) for key in ('pair_id','binding_pair_id')}
 marginals={key:Counter() for key in ('dst','src','expected','value')}
 for c in cases:
  st=empty();conditional_events=[e for e in c['events'] if e['op']==6]
  need(len(conditional_events)==1,'one conditional')
  q=conditional_events[0]
  for e in c['events']:
   if e['op']==6:break
   st,_=transition(st,gold_event(e),exact=True)
  need(st==c['precondition_state'],'precondition metadata')
  actual=read(st,q['src'])==q['expected']
  need(type(c['truth']) is bool and c['truth']==actual and q['gate']==actual,'truth metadata')
  need(any(read(st,i)==q['expected'] for i in range(4)),'global-presence shortcut')
  if not actual:need(any(i!=q['src'] and read(st,i)==q['expected'] for i in range(4)),'false distractor')
  for key in marginals:marginals[key][q[key]]+=1
  for key in pairs:pairs[key][c[key]].append(c)
 for key,groups in pairs.items():
  need(len(groups)==32,'pair group roster')
  for group in groups.values():
   need(len(group)==2 and {c['truth'] for c in group}=={False,True},'pair opposition')
   a,b=group;ea=next(e for e in a['events'] if e['op']==6);eb=next(e for e in b['events'] if e['op']==6)
   if key=='pair_id':
    starta=a['text'].rfind('\n',0,ea['pos'])+1;startb=b['text'].rfind('\n',0,eb['pos'])+1
    need(a['text'][starta:]==b['text'][startb:],'state pair identical suffix')
    need(a['precondition_state']!=b['precondition_state'],'state pair different memories')
   else:
    need(a['precondition_state']==b['precondition_state'],'binding pair identical memory')
    need(ea['src']!=eb['src'] and all(ea[k]==eb[k] for k in ('op','dst','value','expected')),'binding pair changes source only')
 for key,counts in marginals.items():need(len(counts)==4 and set(counts.values())=={16},'operand marginal '+key)
 return {'cases':len(cases),'state_pairs':32,'source_binding_pairs':32,'global_expected_presence':len(cases)}

def pair_diagnostics(cases,rows):
 from collections import defaultdict
 out={}
 for key in ('pair_id','binding_pair_id'):
  groups=defaultdict(list)
  for c,r in zip(cases,rows):groups[c[key]].append(r)
  all_maps=sum(all(r['pred_answers']==r['gold_answers'] for r in rs) for rs in groups.values())
  both_decisions=sum(all(next(e for e in r['gold_events'] if e[1]==6) in r['pred_events'] for r in rs) for rs in groups.values())
  both_sessions=sum(all(r['pred_answers']==r['gold_answers'] and r['pred_states']==r['gold_states'] and r['pred_events']==r['gold_events'] for r in rs) for rs in groups.values())
  out[key]={'pairs':len(groups),'both_conditional_correct':both_decisions,'both_sessions_correct':both_sessions,'conditional_pair_accuracy':both_decisions/len(groups),'session_pair_accuracy':both_sessions/len(groups)}
 return out

def audit(data,result):
 need(set(data)=={'train','dev'},'data splits')
 families=['direct','linked','snapshot','deleted','chain','detached']
 for split,fs,forms in [('train',families[:4],range(2)),('dev',families,(0,4,5))]:
  need(set(data[split])=={f'{f}_form{k}' for f in fs for k in forms},'study data panel roster')
  need(all(len(rows)==64 for rows in data[split].values()),'study data case count')
  for rows in data[split].values():validate_quartets(rows)
 need(result.get('arm') in ('local_read','state_read','local_exact'),'arm')
 need(result.get('seed') in (101,202,303) and result.get('finished_update')==1024,'endpoint identity')
 summaries={}
 for output,split in [('train_probe','train'),('dev','dev')]:
  panels=result[output];need(set(panels)==set(data[split]),'panel roster');summaries[output]={}
  for name,cases in data[split].items():
   p=panels[name];need(set(p)=={'records','metrics','truth_metrics','pair_metrics'},'panel schema')
   rows=p['records'];need([r['case_id'] for r in rows]==[c['case_id'] for c in cases],'exact case roster/order')
   for r,c in zip(rows,cases):validate_record(r,c,result['arm'])
   stats=recompute(rows);compare(stats,p['metrics']);truth={}
   need(set(p['truth_metrics'])=={'0','1'},'truth roster')
   for t in (0,1):
    group=[r for r,c in zip(rows,cases) if int(c['truth'])==t];truth[str(t)]=recompute(group);compare(truth[str(t)],p['truth_metrics'][str(t)])
   paired=pair_diagnostics(cases,rows);compare(paired,p['pair_metrics'])
   summaries[output][name]={'metrics':stats,'truth_metrics':truth,'paired_diagnostics':paired}
 failures=[]
 for name,p in summaries['dev'].items():
  for metric,value in p['metrics'].items():
   if metric!='counts' and value<.95:failures.append([name,'all',metric,value])
  for truth,stats in p['truth_metrics'].items():
   if stats['conditional_decision']<.95:failures.append([name,truth,'conditional_decision',stats['conditional_decision']])
 return {'valid':True,'scope':'Independent saved-record accounting and exact symbolic execution consistency; not checkpoint replay or retraining','arm':result['arm'],'seed':result['seed'],'panels':summaries,'competence_gate':{'passed':not failures,'threshold':.95,'failures':failures,'scope':'All named DEV panels and true/false conditional decisions; three seeds required collectively. Preservation is separate.'}}
def preservation(data,candidate,control):
 """Same-seed casewise answer regressions, not aggregate rescue substitution."""
 audit(data,candidate);audit(data,control)
 need(control['arm']=='local_exact' and candidate['seed']==control['seed'],'preservation control/seed')
 harmed=rescued=0;records=[]
 for panel in data['dev']:
  for a,b in zip(candidate['dev'][panel]['records'],control['dev'][panel]['records']):
   need(a['case_id']==b['case_id'],'paired case identity')
   for q,(gold,ca,cb) in enumerate(zip(a['gold_answers'],a['pred_answers'],b['pred_answers'])):
    damage=cb==gold and ca!=gold;rescue=cb!=gold and ca==gold
    harmed+=damage;rescued+=rescue
    if damage or rescue:records.append(dict(case_id=a['case_id'],query=q,harmed=damage,rescued=rescue))
 return dict(passed=harmed==0,harmed=harmed,rescued=rescued,records=records,scope='DEV paired answers versus same-seed local_exact; no statistical/generalization claim')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('result');p.add_argument('output');a=p.parse_args()
 d=Path(a.data);r=Path(a.result);report=audit(json.loads(d.read_text()),json.loads(r.read_text()));report['data_sha256']=hashlib.sha256(d.read_bytes()).hexdigest();report['result_sha256']=hashlib.sha256(r.read_bytes()).hexdigest();Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
