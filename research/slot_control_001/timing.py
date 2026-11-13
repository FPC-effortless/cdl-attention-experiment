"""Exploratory complete fresh-query sessions, shared tokenization and step loop."""
import argparse,json,time,sys
from pathlib import Path
from run import ROOT,STRESS,verify,setup,load_model,sha,write,validate_closure
from stress_data import VOCAB,encode

def tokens_only(world,query):
    ids=[VOCAB['<bos>']]
    for key,value in world['writes']:
        ids += [VOCAB['set'],VOCAB[f'r{key}'],VOCAB[f'v{value}'],VOCAB[';']]
    return ids+[VOCAB['<nop>']]*4+[VOCAB['ask'],VOCAB[f'r{query}'],VOCAB['?']]

def main(source):
    address=verify(source); validate_closure(source)
    for path in [STRESS/'results/summary.json',ROOT/'results/summary.json']:
        rows=json.loads(path.read_text())
        learned=[r for r in rows if r['arm'] in ('supervised','slot_store')]
        if len(learned)!=30 or any(r['accuracy']<.95 or r['complete_world_accuracy']<.90 for r in learned):
            raise RuntimeError('Reliability screen not met; no timing comparison')
    torch,np,adapter,SlotStore=setup(address)
    worlds=[w for w in json.loads((STRESS/'worlds.json').read_text()) if w['family'].endswith('-s0') and w['updates'] in (1,16,256)]
    results=[]
    with torch.inference_mode():
        for seed in (101,202,303):
            original_path=address/'runs'/f'supervised_{seed}'/'checkpoint.pt'
            original_hash=json.loads((STRESS/'freeze.json').read_text())['checkpoints'][f'supervised_{seed}']
            if sha(original_path)!=original_hash:raise RuntimeError('Original checkpoint changed')
            csm=adapter.RecordRewrite(adapter.Config(vocab_size=len(VOCAB)))
            csm.load_state_dict(torch.load(original_path,map_location='cpu',weights_only=False)['model']);csm.eval()
            slot=load_model(seed,torch,SlotStore)
            models={'csm':csm,'slot':slot}
            def session(model,world,query):
                tick=time.perf_counter_ns()
                ids=torch.tensor([tokens_only(world,query)],dtype=torch.long)
                state=model.init_state(1,'cpu')
                for token in ids.unbind(1):logits,state=model.step(token,state)
                prediction=int(logits.argmax(-1).item())
                return prediction,time.perf_counter_ns()-tick
            for world in worlds:
                # Deterministic rotating identity, not selected by model correctness.
                query=world['block']%4;expected=encode(world,query)['answer']
                if tokens_only(world,query)!=encode(world,query)['input_ids']:raise RuntimeError('Encoding mismatch')
                # Verify common loop predictions agree with each canonical sequence implementation.
                ids=torch.tensor([tokens_only(world,query)])
                canonical_csm=adapter.build_model('supervised',seed,len(VOCAB))
                canonical_csm.load_state_dict(csm.state_dict());canonical_csm.eval()
                original_logits,_=canonical_csm.sequence(ids)
                slot_logits,_=slot.sequence(ids)
                for name,model in models.items():
                    prediction,_=session(model,world,query)
                    ref=original_logits if name=='csm' else slot_logits
                    if prediction!=int(ref.argmax(-1)) or prediction!=expected:raise RuntimeError('Warmup prediction mismatch')
                for rep in range(3):
                    order=('csm','slot') if (world['block']+rep+seed)%2 else ('slot','csm')
                    for name in order:
                        pred,elapsed=session(models[name],world,query)
                        if pred!=expected:raise RuntimeError('Timing prediction mismatch')
                        results.append({'seed':seed,'block':world['block'],'regime':world['regime'],'updates':world['updates'],
                            'query':query,'rep':rep,'order':list(order),'model':name,'elapsed_ns':elapsed,'prediction':pred,'answer':expected})
    write(ROOT/'results/timing.json',{'measurements':results,'threads':1,'batch':1,'repetitions':3,'warmup':1,
        'torch':torch.__version__,'numpy':np.__version__,'freeze_sha256':sha(ROOT/'freeze.json'),
        'kind':'exploratory fresh-query full-session wall time; checkpoint loading excluded; no concurrent fit',
        'csm_retained_float32_tensor_bytes':392*4,'slot_retained_float32_tensor_bytes':320*4,
        'memory_limit':'logical persistent tensors only; excludes Python metadata, weights, inputs and peak temporaries'})
    print('TIMING COMPLETE',len(results),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();main(a.source.resolve())
