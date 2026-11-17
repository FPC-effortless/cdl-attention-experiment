"""Prospective complete-gate aggregation. Does not select checkpoints."""
FIELDS=['query_accuracy','final_map_accuracy','final_state_accuracy','complete_session_accuracy','exact_event_precision','exact_event_recall']
def gates(result):
 panels={split:{name:{f:(panel['metrics'].get(f) is not None and panel['metrics'][f]>=.95) for f in FIELDS} for name,panel in result[split].items()} for split in ['dev','train_probe']}
 return {'qualified':all(v for split in panels.values() for panel in split.values() for v in panel.values()),'criterion':'>=0.95 each field each condition each seed, no aggregation rescue','panels':panels}
