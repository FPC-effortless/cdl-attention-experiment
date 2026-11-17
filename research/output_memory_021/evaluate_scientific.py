"""Scientific resolved-all-records evaluator; frozen raw calibration is unchanged."""
"""021 generated-answer sessions; no gold prior answer is fed at evaluation."""
import json,time
from pathlib import Path
import torch
from conditioner import FrozenSession
from batch_conditioner import BatchedFrozenSession,SerialFrozenBatch
from data_adapter import padded,turns,render_target,extract_payload
from resolved_state import ResolvedExactState as ExactState,ADAPTER_VERSION
from heads import generate_ar,generate_diffusion,decode_task

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()

def evaluate_batch(model,head,records,kind='ar',steps=4,condition='native',intervention='full',length=64,seed=0,target_renderer=render_target,serial_reference=False,software_single=False):
    if not records or (len(records)%2 and not (software_single and len(records)==1 and intervention!='swapped')):raise ValueError('complete paired groups required')
    for i in range(0,len(records)-1,2):
        if records[i]['pair_id']!=records[i+1]['pair_id']:raise ValueError('adjacent paired groups required')
    if condition not in ('native','exact'):raise ValueError('condition')
    if intervention not in ('full','zero','swapped'):raise ValueError('intervention')
    if kind not in ('ar','diffusion'):raise ValueError('head kind')
    device=next(head.parameters()).device;head.eval();torch.set_num_threads(1)
    sync(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    sessions=[ExactState() for r in records] if condition=='exact' else None
    native=(SerialFrozenBatch if serial_reference else BatchedFrozenSession)(model,len(records)) if condition=='native' else None
    streams=[list(turns(r)) for r in records]
    if len({len(x) for x in streams})!=1:raise ValueError('query count mismatch')
    rows=[{'id':r['id'],'pair_id':r['pair_id'],'variant':r['variant'],'answers':[]} for r in records]
    condition_seconds=head_seconds=0.;calls=positions=0;head_receipts=[]
    with torch.inference_mode():
        for q in range(len(streams[0])):
            sync(device);t=time.perf_counter();features=[]
            chunks=[stream[q][1] for stream in streams]
            if condition=='native':
                native.consume(chunks);features=padded(native.feature())
            else:
                for session,chunk in zip(sessions,chunks):session.consume(chunk)
                features=torch.cat([session.feature(device) for session in sessions])
            if intervention=='zero':features=torch.zeros_like(features)
            elif intervention=='swapped':features=features.reshape(-1,2,features.shape[-1]).flip(1).reshape_as(features).clone()
            sync(device);condition_seconds+=time.perf_counter()-t;t=time.perf_counter()
            ids=[f"{r['id']}:q{q}" for r in records]
            if kind=='ar':raw,receipt=generate_ar(head,features,length,ids,seed)
            else:raw,receipt=generate_diffusion(head,features,length,steps,ids,seed)
            sync(device);head_seconds+=time.perf_counter()-t
            calls+=receipt['head_calls'];positions+=receipt['position_evaluations'];head_receipts.append({'query_index':q,**receipt})
            accepted=[]
            for i,(stream,row) in enumerate(zip(streams,rows)):
                _,_,answer=stream[q];generated,status=decode_task(raw[i])
                expected=answer['target'] if target_renderer is None else target_renderer(answer)
                expected=expected.encode('utf8')
                row['answers'].append({'query_index':q,'kind':answer['kind'],'identity':answer['identity'],'target':expected.decode('utf8'),'prediction':generated.decode('utf8',errors='replace'),'prediction_bytes':list(generated),'raw_tokens':raw[i].tolist(),'termination':status,'correct':status=='terminated' and generated==expected,'payload_target':answer['target'],'payload_prediction':extract_payload(generated.decode('utf8',errors='replace')),'payload_correct':status=='terminated' and extract_payload(generated.decode('utf8',errors='replace'))==answer['target']})
                # Common attempted-byte feedback policy, including malformed/truncated partial bytes.
                # No gold answer, no generated control token reaches learned conditioner.
                accepted.append(generated+b'\n')
            if condition=='native':native.consume(accepted)
    sync(device);elapsed=time.perf_counter()-started
    for row in rows:
        row['session_correct']=all(a['correct'] for a in row['answers'])
        row['payload_session_correct']=all(a['payload_correct'] for a in row['answers'])
    return {'batch_id':'|'.join(r['id'] for r in records),'batch_size':len(records),'rows':rows,'kind':kind,'steps':steps if kind=='diffusion' else None,'condition':condition,'exact_adapter_version':ADAPTER_VERSION if condition=='exact' else None,'exact_resolution':[s.receipt() for s in sessions] if sessions is not None else [],'intervention':intervention,'seconds':elapsed,'conditioning_seconds':condition_seconds,'head_seconds':head_seconds,'head_calls':calls,'position_evaluations':positions,'head_receipts':head_receipts,'output_capacity':length,'sampling_seed':seed,'cpu_threads':torch.get_num_threads(),'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'feedback_policy':'always feed attempted byte prefix before EOS/invalid token plus newline exactly once, even on rejected output; no authoritative writes','scope':'whole streaming batch (not per-session latency); generated feedback included; model load and output-file serialization excluded; conditioning subtotal excludes generated-byte feedback, included in total'}

def evaluate_pair(model,head,records,**kwargs):
    if len(records)!=2:raise ValueError('one pair required')
    return evaluate_batch(model,head,records,**kwargs)

def run(model,head,records,output,batch_size=32,**kwargs):
    if batch_size<2 or batch_size%2:raise ValueError('even paired batch size')
    pairs={}
    for r in records:pairs.setdefault(r['pair_id'],[]).append(r)
    ordered=[]
    for pair in pairs.values():
        if len(pair)!=2:raise ValueError('pair roster')
        ordered.extend(sorted(pair,key=lambda r:r['variant']))
    with Path(output).open('w') as f:
        for start in range(0,len(ordered),batch_size):
            result=evaluate_batch(model,head,ordered[start:start+batch_size],**kwargs)
            f.write(json.dumps(result)+'\n');f.flush()
