"""Software calibration only: random fixtures, no scientific head fit or TEST."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,time,subprocess,sys
from pathlib import Path
import torch
from train_heads import train,runtime,configure

def save(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--generator',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();configure()
 assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
 receipt={'scope':'random software optimizer/timing fixtures, no scientific outcome','status':'running','runtime':runtime(),'training':{},'restart':{}}
 save(out/'head_training_calibration.json',receipt)
 try:
  g=torch.Generator().manual_seed(21021);features=torch.randn(128,392,generator=g);targets=torch.full((128,64),257,dtype=torch.long)
  for i in range(128):
   b=f'The current code is {i%1000:03d}.'.encode();targets[i,:len(b)]=torch.tensor(list(b));targets[i,len(b)]=256
  cache=out/'random_fixture_cache.pt';torch.save({'features':features,'exact_features':features.clone(),'targets':targets,'ids':[f'software-{i}' for i in range(128)],'bindings':{'scope':'random fixture only','no_scientific_data':True}},cache)
  total=0
  for mode in ('ar','diffusion'):
   r=train(cache,out/(mode+'_timing'),mode,'learned',11,20,batch=128,device='cuda');total+=20
   steps=[json.loads(x)['step_elapsed_before_trace_write_seconds'] for x in (out/(mode+'_timing')/'updates.jsonl').read_text().splitlines()]
   receipt['training'][mode]={'full20updates':r,'step_seconds':steps,'mean_seconds_per_update_including_end_checkpoint':r['elapsed_seconds']/20}
   train(cache,out/(mode+'_full'),mode,'learned',11,4,batch=4,width=16,layers=1,device='cuda')
   train(cache,out/(mode+'_partial'),mode,'learned',11,4,batch=4,width=16,layers=1,device='cuda',stop=2)
   train(cache,out/(mode+'_resumed'),mode,'learned',11,4,batch=4,width=16,layers=1,device='cuda',resume=out/(mode+'_partial')/'checkpoint.pt');total+=8
   x=torch.load(out/(mode+'_full')/'checkpoint.pt',map_location='cpu',weights_only=True);y=torch.load(out/(mode+'_resumed')/'checkpoint.pt',map_location='cpu',weights_only=True)
   assert equal(x,y),'GPU exact restart mismatch';receipt['restart'][mode]='exact full checkpoint equality';save(out/'head_training_calibration.json',receipt)
  cmd=[sys.executable,'benchmark_adapters.py','--checkpoint',a.checkpoint,'--source',a.source,'--generator',a.generator,'--output',str(out/'adapter'),'--device','cuda','--batch-size','32']
  process=subprocess.run(cmd,capture_output=True,text=True,timeout=1200);(out/'adapter.stdout').write_text(process.stdout);(out/'adapter.stderr').write_text(process.stderr)
  if process.returncode:raise RuntimeError('adapter benchmark failed; retained')
  adapter=json.loads((out/'adapter'/'calibration.json').read_text());assert adapter['status']=='passed' and len(adapter['measurements'])==32
  receipt.update(status='passed',software_optimizer_updates=total,elapsed_seconds=time.perf_counter()-start);save(out/'head_training_calibration.json',receipt)
  result={'metrics':{'scientific_fits':0,'protected_test_examples':0,'software_optimizer_updates':56,'adapter_measurements':32,'restart_arms':2},'runtime':runtime(),'scope':'software calibration only','elapsed_seconds':time.perf_counter()-start}
  save(out/'result.json',result)
 except BaseException as e:
  receipt.update(status='failed',failure={'type':type(e).__name__,'message':str(e)},elapsed_seconds=time.perf_counter()-start);save(out/'head_training_calibration.json',receipt);raise
if __name__=='__main__':main()
