"""Matched natural-text read amplitudes on the 002 article panel."""
import argparse
from torch.nn import functional as F
from common import Path, np, torch, Readout, Runtime, corpus, read, save, model_paths, sequence
from fit import CONFIGS

@torch.inference_mode()
def run(a):
    torch.set_num_threads(1)
    base,head=model_paths(a.parents,'cog_revision')
    original=Runtime.load(base,head)
    heads={'parent':original.head}
    for name in CONFIGS:
        if CONFIGS[name][0]=='cog_revision':
            obj=Readout(original.model.c.width,True)
            obj.load_state_dict(torch.load(a.out/(name+'.pt'),weights_only=True))
            heads[name]=obj.eval()
    arr,off=corpus(a.data,'test')
    ratios={n:[] for n in heads}
    for idx in np.flatnonzero(np.diff(off)>1024):
        start=int(off[idx])
        h,_=sequence(original.model,arr[start:start+1024].tolist())
        for n,obj in heads.items():
            z=h+obj.up(F.gelu(obj.down(h)))
            read,_=obj.bank(z[None])
            ratios[n].append(read[0].norm(dim=-1)/z.norm(dim=-1))
    result={}
    for n,rr in ratios.items():
        r=torch.cat(rr)
        result[n]=dict(tokens=r.numel(),median=r.median().item(),p95=r.quantile(.95).item())
    save(a.out/'usage.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parents','data','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
