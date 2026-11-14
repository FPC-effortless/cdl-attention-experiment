"""Post-execution descriptive state checks; no model calls or decoder fitting."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent
run=next((root/'runs').iterdir())
records=[json.loads(line) for line in (run/'output/predictions.jsonl').open()]
states=np.load(run/'output/state_traces.npz',allow_pickle=False)['states']
lookup={(r['arm'],r['seed'],r['heads'],r['zero_reads'],r['full_gain'],r['key'],r['old'],r['new']):r for r in records}
assert states.shape==(13824,3,32)
assert [r['trace_index'] for r in records]==list(range(13824))
head_diffs=[];state_diffs=[]
for r in records:
 if r['heads']!='normal':continue
 s=lookup[r['arm'],r['seed'],'categorical',r['zero_reads'],r['full_gain'],r['key'],r['old'],r['new']]
 head_diffs.append(max(abs(a-b) for a,b in zip(r['logits'],s['logits'])))
 state_diffs.append(float(np.max(np.abs(states[r['trace_index']]-states[s['trace_index']]))))
summary={'post_hoc':True,'normal_categorical_pairs':len(head_diffs),'max_head_regime_logit_difference':max(head_diffs),'max_head_regime_state_difference':max(state_diffs),
 'max_after_set_to_prequery_change':float(np.max(np.abs(states[:,0]-states[:,1]))),'states_shape':list(states.shape),'contrasts':[]}
for arm in ('csm','pointer_slot'):
 for seed in (101,202,303):
  for zero in (False,True):
   for full in (False,True):
    diffs=[]
    for r in records:
     if (r['arm'],r['seed'],r['heads'],r['zero_reads'],r['full_gain'],r['panel'])!=(arm,seed,'normal',zero,full,'overwrite'):continue
     b=lookup[arm,seed,'normal',zero,full,r['key'],None,r['new']]
     diffs.append(np.max(np.abs(states[r['trace_index']]-states[b['trace_index']]),axis=1))
    summary['contrasts'].append({'arm':arm,'seed':seed,'zero_reads':zero,'full_gain':full,'max_absolute_state_difference_vs_baseline':np.max(diffs,axis=0).tolist()})
(root/'TRACE_ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
