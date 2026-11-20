"""Bounded random-head software cost fixtures, not scientific outcome evaluation."""
import argparse,importlib.util,json,time,os,platform,hashlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import torch
from conditioner import load_conditioner
from batch_conditioner import BatchedFrozenSession,SerialFrozenBatch
from heads import OutputHead
from cache_train import cache
from evaluate_sessions import evaluate_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--generator',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--allow-nonadmitted-cpu',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(21021);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision("highest")
    spec=importlib.util.spec_from_file_location('software_fixture_generator',a.generator);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);receipt={'scope':'random output heads and constructed software fixtures only','device':a.device,'batch_size':a.batch_size,'measurements':[],'status':'running'}
    def save():
        tmp=out/'calibration.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(out/'calibration.json')
    save()
    try:
        if a.batch_size<2 or a.batch_size%2:raise ValueError('even batch size >=2 required')
        device=torch.device(a.device)
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' and torch.cuda.is_available() else None
        admitted=device.type=='cuda' and gpu is not None and 'T4' in gpu
        receipt['runtime']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':gpu,'device':str(device),'gpu_memory_bytes':torch.cuda.get_device_properties(device).total_memory if gpu else None,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'matmul_precision':torch.get_float32_matmul_precision(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'python':platform.python_version(),'admitted_gpu_calibration':admitted}
        receipt['expected_counts']={'evaluation_batches':32,'evaluation_sessions':16*(1+a.batch_size),'evaluation_queries':96*(1+a.batch_size),'cache_sessions':a.batch_size,'cache_targets':6*a.batch_size};save()
        if not admitted and not (a.allow_nonadmitted_cpu and device.type=='cpu'):raise RuntimeError('admitted calibration requires available CUDA Tesla T4; CPU only with explicit nonadmitted flag')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):raise RuntimeError('deterministic CUBLAS workspace required')
        model,bindings=load_conditioner(a.checkpoint,a.source,a.device);receipt['conditioner']=bindings
        head=OutputHead(392).to(a.device).eval()
        parity={'status':'running','feature_atol':1e-6,'feature_rtol':1e-5};receipt['gpu_parity']=parity;save()
        with torch.inference_mode():
            serial=SerialFrozenBatch(model,2);batched=BatchedFrozenSession(model,2)
            maxdiff=0.
            for chunks in [[b'The current code for Mira is ABC.\n',b'The current code for Mira is DEF.\n'],[b'What code does Mira currently have?\nAnswer: ',b''],[b'',b'What code does Mira currently have?\nAnswer: ']]:
                serial.consume(chunks);batched.consume(chunks)
                sf,bf=serial.feature(),batched.feature();maxdiff=max(maxdiff,float((sf-bf).abs().max()))
                if not torch.allclose(sf,bf,atol=1e-6,rtol=1e-5):raise RuntimeError('GPU serial/batched frozen feature divergence')
            parity['feature_max_abs']=maxdiff;save()
            token=torch.randint(0,259,(2,64),device=a.device);condition=batched.feature()
            full=head(token,condition,causal=True);kv=None;pieces=[]
            for step in range(64):
                logits,kv=head.ar_step(token[:,step:step+1],condition,kv);pieces.append(logits)
            incremental=torch.cat(pieces,1)
            parity['cached_ar_logits_max_abs']=float((full-incremental).abs().max())
            parity['cached_ar_64_logits_allclose']=bool(torch.allclose(full,incremental,atol=1e-6,rtol=1e-5));save()
            if not parity['cached_ar_64_logits_allclose']:raise RuntimeError('GPU cached/uncached64 AR logits divergence')
            fixture=g.episode_pair('software-parity',910021,64)
            for kind in ['ar','diffusion']:
                batch=evaluate_batch(model,head,fixture,kind=kind,steps=4)
                individual=[evaluate_batch(model,head,[row],kind=kind,steps=4,software_single=True,serial_reference=True)['rows'][0] for row in fixture]
                evidence=out/f'gpu_parity_{kind}_sessions.json'
                evidence.write_text(json.dumps({'batch_rows':batch['rows'],'serial_rows':individual},indent=2))
                parity[kind+'_raw_evidence_sha256']=hashlib.sha256(evidence.read_bytes()).hexdigest()
                same=batch['rows']==individual
                parity[kind+'_whole_session_raw_status_equal']=same;save()
                if not same:raise RuntimeError(f'GPU {kind} whole session batch/serial divergence')
            parity['status']='passed';save()
        training=[r for i in range(a.batch_size//2) for r in g.episode_pair('train',900000+i,256)]
        c=cache(model,training,out/'software_cache.pt',conditioner_receipt=bindings,batch_size=a.batch_size)
        receipt['cache_calibration']=c;save()
        for delay in [256,2048]:
            fixtures=[r for i in range(a.batch_size//2) for r in g.episode_pair('software',900000+i,delay)]
            for condition in ['native','exact']:
                for kind,steps in [('ar',4),('diffusion',4),('diffusion',8),('diffusion',16)]:
                    for batch in [1,a.batch_size]:
                        result=evaluate_batch(model,head,fixtures[:batch],condition=condition,kind=kind,steps=steps,software_single=batch==1)
                        # Do not publish random fixture accuracy as research findings.
                        receipt['measurements'].append({k:v for k,v in result.items() if k!='rows'}|{'delay':delay});save()
        receipt['status']='passed';save()
    except Exception as e:
        receipt['status']='failed';receipt['failure']={'type':type(e).__name__,'message':str(e)};save();raise
if __name__=='__main__':main()
