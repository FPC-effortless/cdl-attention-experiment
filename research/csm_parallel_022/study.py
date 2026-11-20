"""Prospective CPU scientific seed runner. Requires separate native admission."""
import argparse,dataclasses,hashlib,json,time,random
from pathlib import Path
import torch
from models import SentenceParser,Config,OPS
from train import train,configure,sha
from runtime022 import fingerprint
from data import validate_roster,external_events,payload_text
from evaluate import run,session,commands,score,audit_record
from executor import Executor,build

def loadmodel(checkpoint):
 cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
 model=SentenceParser(Config(**cp['bindings']['config']['model']));model.load_state_dict(cp['model'],strict=True);model.eval();return model,cp

def rows_equal(a,b):
 keys=['id','pair_id','variant','actions','typed_commands','registry','errors','answers','rendered_answers','predicted_guards','final_state']
 return len(a)==len(b) and all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))

def run_seed(data_dir,out,seed,updates=2048,batch=32):
 if seed not in [11,23,37] or updates!=2048 or batch!=32:raise ValueError('unreviewed seed/budget')
 configure();start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False);data_dir=Path(data_dir)
 train_records=[json.loads(x) for x in (data_dir/'train.jsonl').read_text().splitlines()];validate_roster(train_records)
 examples=[{'id':f'{r["id"]}:{i}','text':e['text'],'labels':e['labels']} for r in train_records for i,e in enumerate(r['events'])]
 source=out/'train_instructions.jsonl';source.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in examples))
 panels={p.stem:[json.loads(x) for x in p.read_text().splitlines()] for p in sorted(data_dir.glob('dev_*.jsonl'))}
 if len(panels)!=7:raise ValueError('expected seven axes')
 for rows in panels.values():validate_roster(rows)
 totals={'scientific_fits':0,'optimizer_updates':0,'instruction_exposures':0,'primary_sessions':0,'primary_queries':0,'serial_parity_sessions':0,'checkpoint_replay_sessions':0,'protected_test_examples':0}
 results={'seed':seed,'training':{},'panels':{},'timing':[],'engineered':{},'data_sha256':{p.name:sha(p) for p in data_dir.glob('*.jsonl')},'scope':'supervised constructed instruction parser plus exact memory; not native CSM equivalence'}
 compile_start=time.perf_counter();ex=Executor(build(out/'build'));results['executor_construction_seconds']=time.perf_counter()-compile_start
 for arm in ['parallel','gru']:
  result=train(source,out/arm,arm,seed,updates,batch=batch);results['training'][arm]=result
  totals['scientific_fits']+=1;totals['optimizer_updates']+=result['optimizer_updates'];totals['instruction_exposures']+=result['instruction_exposures']
  model,cp=loadmodel(out/arm/'checkpoint.pt')
  actual_parameters=sum(cp['model'][name].numel() for name,_ in model.named_parameters())
  if actual_parameters!=cp['parameters'] or actual_parameters!=result['parameters']:raise ValueError('checkpoint tensor parameter reconciliation')
  results['training'][arm]['parameters']=actual_parameters
  traces=[json.loads(x) for x in (out/arm/'updates.jsonl').read_text().splitlines()]
  if cp['updates']!=updates or len(traces)!=updates or sum(x['instructions'] for x in traces)!=result['instruction_exposures']:raise ValueError('actual checkpoint/trace budget mismatch')
  results['training'][arm]['trace_sha256']=sha(out/arm/'updates.jsonl')
  results['training'][arm]['loss_quarter_means']=[sum(x['loss'] for x in traces[q*512:(q+1)*512])/512 for q in range(4)]
  results['panels'][arm]={}
  for axis,records in panels.items():
   prefix=out/(arm+'_'+axis)
   serial=run(model,records,str(prefix)+'_serial.json',chunk=1,parallel=False,threads=1,executor=ex)
   primary=run(model,records,str(prefix)+'_primary.json',chunk=32,parallel=True,threads=4,reference_rows={r['id']:r for r in serial['rows']},executor=ex)
   restored,_=loadmodel(out/arm/'checkpoint.pt')
   replay=run(restored,records,str(prefix)+'_replay.json',chunk=32,parallel=True,threads=4,reference_rows={r['id']:r for r in primary['rows']},executor=ex)
   if not rows_equal(primary['rows'],replay['rows']):raise RuntimeError('exact checkpoint replay failed')
   results['panels'][arm][axis]={'scores':primary['scores'],'serial_seconds':serial['complete_session_seconds'],'primary_seconds':primary['complete_session_seconds'],'replay_seconds':replay['complete_session_seconds'],'exact_replay':True}
   totals['primary_sessions']+=len(records);totals['primary_queries']+=primary['scores']['queries'];totals['serial_parity_sessions']+=len(records);totals['checkpoint_replay_sessions']+=len(records)
   # Prospectively fixed first two complete pairs/axis; rotating order, not outcome-selected.
   subset=records[:4];validate_roster(subset)
   schedules=[(1,False,1),(8,False,1),(32,False,1),(128,False,1),(32,True,4)]
   for rep in range(3):
    order=schedules[rep:]+schedules[:rep]
    for chunk,parallel,threads in order:
     elapsed=0.;raw=[]
     for record in subset:
      row,_=session(model,record,ex,chunk,parallel,threads);audit_record(record,row);raw.append(row);elapsed+=row['cost']['complete_session_seconds']
      expected=next(x for x in primary['rows'] if x['id']==record['id'])
      if not rows_equal([row],[expected]):raise RuntimeError('timed schedule parity')
     results['timing'].append({'arm':arm,'axis':axis,'repeat':rep,'chunk':chunk,'parallel':parallel,'threads':threads,'sessions':len(subset),'seconds':elapsed,'raw_rows':raw})
 schedules=[[json.loads(x)['sampled_ids_sha256'] for x in (out/arm/'updates.jsonl').read_text().splitlines()] for arm in ['parallel','gru']]
 if schedules[0]!=schedules[1]:raise RuntimeError('paired sampling schedules differ')
 results['matched_sampling_schedules']=True
 # Actual supplied-program reference, separately labeled; no neural capability claim.
 for axis,records in panels.items():
  entries=[]
  for r in records:
   methods={}
   for label,parallel_mode,threads in [('serial',False,1),('parallel',True,4)]:
    t=time.perf_counter();actions=[{'op':OPS[e['labels']['op']],'valid':True,'spans':e['labels']['spans']} for e in r['events']]
    ops,n,banks,errors,registry=commands(external_events(r),actions)
    value=ex.execute(ops,n,bank_ids=banks,threads=threads,parallel=parallel_mode)
    rendered=[None if a==-2 else ('No code is currently assigned.' if a is None else 'The current code is '+payload_text(a)+'.') for a in value['answers']]
    methods[label]={**value,'rendered_answers':rendered,'full_supplied_program_session_seconds':time.perf_counter()-t}
   if methods['serial']['state']!=methods['parallel']['state'] or methods['serial']['answers']!=methods['parallel']['answers']:raise RuntimeError('engineered exact parity')
   entries.append({'id':r['id'],**methods})
  results['engineered'][axis]=entries
 results['metrics']=totals;results['runtime']=fingerprint();results['wall_seconds']=time.perf_counter()-start
 results['adequacy_observations']={'model_parameters_total':sum(x['parameters'] for x in results['training'].values()),'optimizer_updates':totals['optimizer_updates'],'training_instructions':len(examples),'instruction_exposures':totals['instruction_exposures']}
 (out/'result.json').write_text(json.dumps(results,indent=2))
 files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
 (out/'evaluation_index.json').write_text(json.dumps({'files':files},indent=2));return totals
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=2048);p.add_argument('--batch',type=int,default=32);a=p.parse_args();print(json.dumps(run_seed(a.data,a.out,a.seed,a.updates,a.batch)))
