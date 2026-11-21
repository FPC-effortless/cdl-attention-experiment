import argparse
from common import Path, torch, Runtime, save, model_paths, build_docs
from fit import CONFIGS
from architecture import mix
from panels import natural
from torch.nn import functional as F


def system(a,name):
    arm=CONFIGS[name][0]
    base,original=model_paths(a.parents,arm)
    return Runtime.load(base,a.out/(name+'.pt'))


@torch.inference_mode()
def probes(a,name):
    net=system(a,name)
    head=net.head
    arm=CONFIGS[name][0]
    results=[]
    for i,doc in enumerate(build_docs(a.data,'test')):
        h,ids,qs=torch.load(a.cache/(arm+'-test-'+str(i)+'.pt'),weights_only=True)['long']
        state=None
        previous=0
        for end,queries in qs.items():
            if head.bank:
                for chunk in h[previous:end].split(256):
                    block=chunk[None]
                    z=block+head.up(F.gelu(head.down(block)))
                    _,state=head.bank(z,state)
            previous=end
            for j,((qh,qi),_) in enumerate(queries):
                q=doc['questions'][j]
                z=qh[None]+head.up(F.gelu(head.down(qh[None])))
                residual,_=head.bank(z,state) if head.bank else (torch.zeros_like(z),None)
                z=z[:,-1:]
                residual=residual[:,-1:]
                def get(v,index):
                    logp=F.log_softmax(F.linear(v,net.model.emb.weight),dim=-1)
                    p=mix(logp,index,head.gate(v).squeeze(-1))[0,0]
                    return dict(top=int(p.argmax()),odds=float(p[q['target']]-p[q['foil']]))
                idx=qi[-1:].reshape(1,1)
                results.append(dict(family=doc['family'],flip=doc['flip'],kind=q['kind'],who=q['who'],
                    delay=end-len(doc['prefix']),gold=q['target'],foil=q['foil'],lookup=int(idx),
                    full=get(z+residual,idx),masked=get(z,idx),no_lookup=get(z+residual,torch.full_like(idx,-1)),
                    ratio=float(residual.norm()/z.norm().clamp_min(1e-9))))
    save(a.out/(name+'.probes.json'),results)
    save(a.out/(name+'.text.json'),{s:natural(net,a.data,s) for s in ('test','transfer')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cache','data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',choices=CONFIGS,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    probes(a,a.name)
