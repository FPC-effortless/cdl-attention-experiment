"""017 strict roster-bound gate. Missing data never qualifies."""
import math
FIELDS=['query_accuracy','final_map_accuracy','final_state_accuracy','complete_session_accuracy','exact_event_precision','exact_event_recall']
def gates(result,required_panels,required_seed):
 if result.get('seed')!=required_seed:raise ValueError('seed mismatch')
 if set(required_panels)!={'dev','train_probe'} or any(not v for v in required_panels.values()):raise ValueError('nonempty split rosters required')
 panels={}
 for split,roster in required_panels.items():
  if type(roster) is not dict:raise ValueError('exact per-panel case IDs required')
  names=list(roster)
  if len(names)!=len(set(names)) or set(result.get(split,{}))!=set(names):raise ValueError('panel roster mismatch')
  panels[split]={}
  for name in names:
   panel=result[split][name];expected=roster[name];actual=[r['case_id'] for r in panel.get('records',[])]
   if not expected or len(expected)!=len(set(expected)) or actual!=expected:raise ValueError('case roster mismatch')
   metrics=panel['metrics'];panels[split][name]={}
   if metrics.get('counts',{}).get('cases',0)!=len(expected):raise ValueError('empty panel')
   for f in FIELDS:
    v=metrics.get(f)
    if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('missing/nonfinite metric')
    panels[split][name][f]=v>=.95
 return {'qualified':all(v for split in panels.values() for panel in split.values() for v in panel.values()),'criterion':'>=0.95 each field each condition each seed; no aggregation rescue','panels':panels}
