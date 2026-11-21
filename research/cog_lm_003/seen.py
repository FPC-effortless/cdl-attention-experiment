"""Measure fit on the actual training episodes using intact lookup and recurrent state."""
import argparse
from common import Path, torch, save, build_docs
from fit import CONFIGS,sparse
from assess import system


@torch.no_grad()
def run(a):
    torch.set_num_threads(1)
    rows=[]
    for name in ('route_revision','floor_revision','long_revision','route_cog_control','route_mamba_control'):
        net=system(a,name)
        for i,d in enumerate(build_docs(a.data,'train')):
            h,ids,qs=torch.load(a.cache/(CONFIGS[name][0]+'-train-'+str(i)+'.pt'),weights_only=True)['long']
            for j,q in enumerate(d['questions']):
                qh,qi=qs[len(h)][j][0]
                x=torch.cat([h,qh])
                p,ratio=sparse(net.head,net.model.emb.weight,x,torch.cat([ids,qi]),torch.tensor([len(x)-1]))
                rows.append(dict(name=name,example=i,family=d['family'],flip=d['flip'],gold=q['target'],foil=q['foil'],
                    who=q['who'],kind=q['kind'],delay=d['delay'],top=int(p.argmax()),odds=float(p[0,q['target']]-p[0,q['foil']]),
                    ratio=float(ratio[0])))
    save(a.out/'train_replay.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parents','cache','data','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
