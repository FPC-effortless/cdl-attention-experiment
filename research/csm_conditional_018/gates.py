"""No means rescue failed seeds/panels; false conditional decisions count."""
FIELDS=['query_accuracy','complete_map','final_state','complete_session','conditional_decision','conditional_precision','conditional_recall']
def qualify(result):
 failures=[]
 for panel,entry in result['dev'].items():
  for key in FIELDS:
   if entry['metrics'].get(key,0)<.95:failures.append([panel,key,entry['metrics'].get(key)])
  for truth in ('0','1'):
   value=entry['truth_metrics'][truth]['conditional_decision']
   if value<.95:failures.append([panel,'conditional_decision_truth'+truth,value])
 return dict(pass_all=not failures,failures=failures,scope='individual fit only; all seeds and preservation comparison separately required')
def preservation(candidate,control):
 errors=[]
 if set(candidate['dev'])!=set(control['dev']):raise ValueError('panel mismatch')
 for panel,c in candidate['dev'].items():
  ref={r['case_id']:r for r in control['dev'][panel]['records']}
  if {r['case_id'] for r in c['records']}!=set(ref):raise ValueError('case mismatch')
  for r in c['records']:
   b=ref[r['case_id']]
   if r['gold_answers']!=b['gold_answers']:raise ValueError('gold mismatch')
   for i,gold in enumerate(r['gold_answers']):
    if b['pred_answers'][i]==gold and r['pred_answers'][i]!=gold:errors.append([r['case_id'],i])
 return dict(no_new_wrong_answers=not errors,new_errors=errors)
