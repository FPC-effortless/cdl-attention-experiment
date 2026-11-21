"""Eight rotated batch-one FP32 CPU requests for matched post-training systems."""
import argparse,time
from common import Path, torch, Tokenizer, Runtime, corpus, save, model_paths
from assess import system


@torch.inference_mode()
def run(a):
    torch.set_num_threads(2)
    names=['route_revision','route_cog_control','route_mamba_control','parent_mamba']
    models={k:system(a,k) for k in names[:-1]}
    models['parent_mamba']=Runtime.load(*model_paths(a.parents,'mamba_lookup'))
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    arr,offs=corpus(a.data,'test')
    start=next(int(lo)+32 for lo,hi in zip(offs,offs[1:]) if hi-lo>17000)
    prompt=tok.decode(arr[start:start+1024].tolist())
    rows=[]
    for rep in range(9):
        order=names if rep<5 else names[::-1]
        shift=(rep-1)%4
        for k in order[shift:]+order[:shift]:
            t=time.perf_counter()
            x=torch.tensor([tok.encode(prompt).ids])
            p,state=models[k].advance(x,last_only=True)
            loaded=time.perf_counter()
            out=[]
            for _ in range(32):
                token=p[:,-1].argmax(-1,keepdim=True)
                out.append(token.item())
                p,state=models[k].advance(token,state,last_only=True)
            tok.decode(out)
            stop=time.perf_counter()
            if rep:
                rows.append(dict(name=k,rep=rep,tokens=x.shape[1],encode_prefill_s=loaded-t,
                    decode_s=stop-loaded,request_s=stop-t,ids=out))
    save(a.out/'timing.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
