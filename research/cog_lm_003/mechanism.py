"""Mechanistic controls on the cached features. Scores use the same target and foil."""
import argparse
from common import Path, torch, Runtime, save, model_paths, build_docs
from fit import CONFIGS,sparse

def run(a):
    torch.set_num_threads(1)
    docs=build_docs(a.data,'train')
    arm='cog_revision'
    base,headpath=model_paths(a.parents,arm)
    m=Runtime.load(base,headpath)
    rows=[]
    for name in ('parent','route_revision','floor_revision'):
        source=headpath if name=='parent' else a.out/(name+'.pt')
        m.head.load_state_dict(torch.load(source,weights_only=True))
        for i in (4,5,10,11):
            d=docs[i]
            assert d['delay']==2048
            item=torch.load(a.cache/(arm+'-train-'+str(i)+'.pt'),weights_only=True)
            for short in (True,False):
                h,ids,qs=item['short' if short else 'long']
                j=1
                q=d['questions'][j]
                qh,qi=qs[len(h)][j][0]
                x=torch.cat([h,qh]).requires_grad_(True)
                pos=torch.tensor([len(x)-1])
                p,ratio=sparse(m.head,m.model.emb.weight,x,torch.cat([ids,qi]),pos,short)
                grad,=torch.autograd.grad(p[0,q['target']]-p[0,q['foil']],x)
                rows.append(dict(name=name,short=short,example=i,
                    prefix_feature_gradient=float(grad[:len(d['prefix'])].norm()),
                    query_feature_gradient=float(grad[len(h):].norm()),ratio=float(ratio.detach()[0])))
    save(a.out/'gradient.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cache','data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
