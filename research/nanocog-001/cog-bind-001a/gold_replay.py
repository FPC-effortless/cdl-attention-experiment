"""Diagnostic only: score exact frozen-decoder output after forcing gold clause."""
import argparse
from pathlib import Path
import torch
from run import ARMS, SEEDS, PANELS, Model, put, read, req, sha

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args()
    req((a.out/'RESULTS.json').is_file(),'primary must be sealed')
    for arm in ARMS:
        for seed in SEEDS:
            d=a.out/f'{arm}-{seed}'
            model_file=d/'weights.pt'
            f=read(d/'fit.json');req(sha(model_file)==f['weight_sha'],'weight seal')
            m=Model(arm);m.load_state_dict(torch.load(model_file,weights_only=True,map_location='cpu'));m.eval()
            actor=read(d/'predict.json'); manifest={}
            for panel in PANELS:
                raw=d/(panel+'.pt'); req(sha(raw)==actor[panel]['wire_sha'],'actor seal')
                w=torch.load(raw,weights_only=True,map_location='cpu')
                t=a.data/(panel+'-truth.json'); y=torch.tensor([x['target'] for x in read(t)]).flatten()
                with torch.no_grad():
                    i=torch.arange(len(y)); h=w['weights'].argmax(-1); q=w['q']; v=w['v']
                    native=m.final(torch.cat((q,(v*w['weights'][:,:,None]).sum(1)),dim=-1))
                    own=m.final(torch.cat((q,v[i,h]),dim=-1))
                    gold=m.final(torch.cat((q,v[i,y]),dim=-1))
                    req(torch.allclose(native,w['logits'],atol=1e-5,rtol=1e-6),'native mismatch')
                    req(torch.allclose(own,w['hard'],atol=1e-5,rtol=1e-6),'hard mismatch')
                    req(torch.equal(own[h==y],gold[h==y]),'matching supports differ')
                o=d/(panel+'-gold.pt'); torch.save({'gold':gold},o)
                manifest[panel]={'primary_sha':sha(raw),'gold_sha':sha(o),'truth_sha':sha(t)}
            put(d/'gold.json',manifest)

if __name__=='__main__':main()
