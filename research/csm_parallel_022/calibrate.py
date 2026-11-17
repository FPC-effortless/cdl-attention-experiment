"""Bounded CPU software calibration; random fixtures, no capability outcome."""
import argparse,json,time,hashlib,dataclasses
from pathlib import Path
import torch
from data import episode_pair,regions_pair,OPS,external_events
from models import SentenceParser,Config,infer_chunks,parity_guard
from evaluate import session,commands
from executor import Executor,build
from train import train,configure
from runtime022 import fingerprint

def main(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);configure();start=time.perf_counter()
 records=[r for i in range(2) for r in episode_pair('software-calibration-only',900000+i)]+regions_pair('software-regions-only',900000)+episode_pair('software-cross-only',900000,cross_bank=True)
 examples=[{'id':f'{r["id"]}-{i}','text':e['text'],'labels':e['labels']} for r in records for i,e in enumerate(r['events'])]
 source=out/'software_examples.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in examples))
 measurements={'scope':'software only, constructed high-index fixtures; no model capability result; fixed order single observations for resource calibration, not speed ranking','training':{},'inference':[],'engineered':[],'parity':[]}
 compile_start=time.perf_counter();lib=build(out/'build');ex=Executor(lib);measurements['executor_load_build_seconds']=time.perf_counter()-compile_start;measurements['executor_binary_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
 for arm in ['parallel','gru']:
  measurements['training'][arm]=train(source,out/(arm+'_software'),arm,12345,8,batch=16)
  torch.manual_seed(23456);model=SentenceParser(Config(arm=arm)).eval()
  for record in records:
   base,base_logits=session(model,record,ex,1,False,1)
   for chunk in [1,8,32,128]:
    row,z=session(model,record,ex,chunk,True,4)
    p=parity_guard(base_logits,base['actions'],z,row['actions'])
    for key in ['typed_commands','registry','errors','answers','final_state']:
     if row[key]!=base[key]:raise RuntimeError('end-to-end parity '+key)
    measurements['inference'].append({'arm':arm,'id':record['id'],'events':len(record['events']),'chunk':chunk,**row['cost'],'executor_stats':row['executor_stats']})
    measurements['parity'].append({'arm':arm,'id':record['id'],'chunk':chunk,**p})
 for record in records:
  actions=[{'op':OPS[e['labels']['op']],'valid':True,'spans':e['labels']['spans']} for e in record['events']]
  ops,n,banks,errors,ids=commands(external_events(record),actions)
  reference=ex.execute(ops,n,bank_ids=banks,threads=1,parallel=False)
  for threads,parallel in [(1,False),(1,True),(4,True)]:
   result=ex.execute(ops,n,bank_ids=banks,threads=threads,parallel=parallel)
   if result['answers']!=reference['answers'] or result['state']!=reference['state']:raise RuntimeError('engineered measured parity')
   digest=hashlib.sha256(json.dumps({'answers':result['answers'],'state':result['state']},sort_keys=True).encode()).hexdigest()
   measurements['engineered'].append({'id':record['id'],'threads':threads,'parallel':parallel,'exact_output_sha256':digest,'parity':True,**result['stats']})
 (out/'measurements.json').write_text(json.dumps(measurements,indent=2))
 result={'metrics':{'software_optimizer_updates':16,'parser_timing_cases':64,'engineered_timing_cases':24,'scientific_fits':0,'protected_test_examples':0},'runtime':fingerprint(),'seconds':time.perf_counter()-start,'measurements_sha256':hashlib.sha256((out/'measurements.json').read_bytes()).hexdigest()}
 (out/'result.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(main(a.out)))
