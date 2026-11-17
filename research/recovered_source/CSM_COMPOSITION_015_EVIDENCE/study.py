import json,os,torch
from pathlib import Path
from model import Model,tensor
from evaluate import validate_record,metrics_from_records
from closed_loop import rollout,gold,record,MODES
from open_loop import evaluate,replay

def fixed_actions(r,mode):
 pred={e[0]:e.copy() for e in r['pred_nonnoop']};g={e[0]:e for e in r['gold_nonnoop']}
 if mode=='all_gold_ceiling':return r['gold_nonnoop']
 if mode=='suppress_false':return [e for p,e in pred.items() if p in g]
 for p,a in g.items():
  b=pred.get(p)
  if mode=='positive_actions':pred[p]=a.copy()
  elif mode=='both_identities' and a[1] in (2,3) and b and b[1]==a[1]:b[2:4]=a[2:4]
 return [e for p,e in sorted(pred.items())]

def same_state(a,b):
 return all(torch.equal(i,j) for i,j in zip(a[0],b[0])) and torch.equal(a[1],b[1]) and all(torch.equal(a[2][k],b[2][k]) for k in a[2])
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);B=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);cases=json.loads((B/'closed_cases.json').read_text());chosen={q['case_id'] for qs in cases.values() for q in qs};original={};n=0
 with (O/'open_results.jsonl').open('w') as out:
  for line in (B/'records.jsonl').open():
   row=json.loads(line);out.write(json.dumps(evaluate([row])[0])+'\n');n+=1
   if row['record']['case_id'] in chosen:original[(row['arm'],row['seed'],row['panel'],row['record']['case_id'])]=row['record']
 assert n==30720 and len(original)==3840
 count=0;stats=[]
 with torch.no_grad(),(O/'closed_results.jsonl').open('w') as out:
  for path in sorted((B/'checkpoints').glob('*.pt')):
   arm,seed=path.stem.rsplit('-',1);seed=int(seed);m=Model(arm);m.load_state_dict(torch.load(path,weights_only=True));m.eval();modelcount=0
   for panel,qs in cases.items():
    x=tensor(qs);g=gold(qs);base=rollout(m,x,m.init(len(qs)),g,'original');ordinary=m.segment(x,m.init(len(qs)),trace=True)
    assert torch.equal(base[0],ordinary[0]) and torch.equal(base[1],ordinary[1]) and same_state(base[2],ordinary[2]) and all(torch.equal(i,j) for i,j in zip(base[3],ordinary[3]))
    for mode in MODES:
     result=base if mode=='original' else rollout(m,x,m.init(len(qs)),g,mode)
     assert torch.equal(result[2][1],base[2][1]) and all(torch.equal(result[2][2][k],base[2][2][k]) for k in base[2][2])
     if not m.conditioned:assert torch.equal(result[0],base[0]) and torch.equal(result[4],base[4])
     for i,q in enumerate(qs):
      r=record(q,result,i,arm,mode,base);validate_record(r);orig=original[(arm,seed,panel,q['case_id'])]
      if mode=='original':
       for k in orig:assert r[k]==orig[k],(arm,seed,panel,q['case_id'],k)
      fixed=replay(orig,fixed_actions(orig,mode));closed={k:r[k] for k in fixed};matched=fixed==closed
      if not m.conditioned:assert matched
      metrics=metrics_from_records([r]);basecorrect=[p==g for p,g in zip(orig['query_predictions'],orig['query_targets'])];correct=[p==g for p,g in zip(r['query_predictions'],r['query_targets'])]
      if mode=='all_gold_ceiling':assert metrics['complete_session_accuracy']==1
      out.write(json.dumps({'arm':arm,'seed':seed,'panel':panel,'mode':mode,'record':r,'executed_behavior_metrics':metrics,'matched_subset_open_loop':fixed,'open_closed_exact':matched,'rescued_queries':sum(c and not b for c,b in zip(correct,basecorrect)),'damaged_queries':sum(b and not c for c,b in zip(correct,basecorrect))})+'\n');count+=1;modelcount+=1
   stats.append({'arm':arm,'seed':seed,'mode_histories':modelcount});(O/'progress.json').write_text(json.dumps(stats))
 assert count==19200
 (O/'results.json').write_text(json.dumps({'open_histories':n,'open_modes':9,'closed_histories':3840,'closed_mode_histories':count,'fits':0,'models':stats,'scope':'privileged interventions, not learned repair'}));(O/'metrics.json').write_text(json.dumps({'open_histories':n,'closed_mode_histories':count,'fits':0}))
