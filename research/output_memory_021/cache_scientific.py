"""Scientific resolved-all-records adapter; raw calibration cache remains separate."""
"""Teacher-prefix TRAIN feature cache; never reusable for free-running DEV."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from conditioner import FrozenSession,load_conditioner
from batch_conditioner import BatchedFrozenSession,SerialFrozenBatch
from data_adapter import turns,padded,read_train,render_target
from heads import EOS,PAD
from resolved_state import ResolvedExactState as ExactState,ADAPTER_VERSION

def cache(model,records,output,length=64,conditioner_receipt=None,data_sha256=None,batch_size=32,serial_reference=False):
    torch.set_num_threads(1);started=time.perf_counter();rows=[]
    if not records or any(not r['id'].startswith('train-') for r in records):raise ValueError('TRAIN only')
    if batch_size<1:raise ValueError('positive batch size')
    saved={};resolution_receipts=[]
    for start in range(0,len(records),batch_size):
        group=records[start:start+batch_size]
        native=(SerialFrozenBatch if serial_reference else BatchedFrozenSession)(model,len(group))
        exact=[ExactState() for _ in group];streams=[list(turns(r)) for r in group]
        if len({len(x) for x in streams})!=1:raise ValueError('query count mismatch')
        for q in range(len(streams[0])):
            chunks=[s[q][1] for s in streams];native.consume(chunks)
            features=padded(native.feature()).cpu();prior=[]
            for i,(record,stream,e,chunk) in enumerate(zip(group,streams,exact,chunks)):
                index,_,answer=stream[q];e.consume(chunk)
                target=render_target(answer).encode('utf8')
                if len(target)+1>length:raise ValueError('output length insufficient')
                clean=list(target)+[EOS]+[PAD]*(length-len(target)-1)
                identity=f"{record['id']}:q{index}"
                if identity in saved:raise ValueError('duplicate training ID')
                saved[identity]={'id':identity,'native':features[i:i+1],'exact':e.feature().cpu(),'clean':torch.tensor(clean,dtype=torch.long)}
                prior.append(target+b'\n')
            # Teacher-prefix TRAIN only. Never use this cache for free-running DEV.
            native.consume(prior)
        resolution_receipts.extend(e.receipt() for e in exact)
    # Preserve identical roster ordering independent of batch size.
    rows=[saved[f"{r['id']}:q{q}"] for r in records for q in range(len(r['answers']))]
    payload={'features':torch.cat([r['native'] for r in rows]),'exact_features':torch.cat([r['exact'] for r in rows]),'targets':torch.stack([r['clean'] for r in rows]),'ids':[r['id'] for r in rows], 'bindings':{'conditioner':conditioner_receipt,'data_sha256':data_sha256,'format':'canonical-code-sentence-v1','condition_dim':392,'exact_adapter_version':ADAPTER_VERSION,'output_length':length,'cache_batch_size':batch_size,'serial_reference':serial_reference},'scope':'TRAIN only; earlier canonical answers teacher-forced; current/future targets excluded from features','seconds':time.perf_counter()-started,'exact_resolution':{'adapter_version':ADAPTER_VERSION,'feature_calls':sum(r['feature_calls'] for r in resolution_receipts),'resolved_records':sum(r['resolved_records'] for r in resolution_receipts),'reference_resolution_hops':sum(r['reference_resolution_hops'] for r in resolution_receipts),'feature_seconds':sum(r['feature_seconds'] for r in resolution_receipts)}}
    torch.save(payload,output)
    receipt={k:v for k,v in payload.items() if k not in ('features','exact_features','targets','ids')};receipt.update({'examples':len(rows),'sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest()})
    Path(str(output)+'.json').write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--length',type=int,default=64);a=p.parse_args()
    m,r=load_conditioner(a.checkpoint,a.source,a.device);print(json.dumps(cache(m,read_train(a.data),a.output,a.length,r,hashlib.sha256(Path(a.data).read_bytes()).hexdigest())))
