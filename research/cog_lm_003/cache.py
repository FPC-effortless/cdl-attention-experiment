import argparse
import copy
from common import *


def run(a):
    torch.set_num_threads(a.threads)
    base,_=model_paths(a.parents,a.arm)
    m=backbone(base)
    for split in ('train','test'):
        docs=build_docs(a.data,split)
        for i,doc in enumerate(docs):
            path=a.out/(a.arm+'-'+split+'-'+str(i)+'.pt')
            if path.exists():
                continue
            all_tokens=doc['prefix']+doc['filler']
            # Restart at multiples of 128 on the short control, with the same data order.
            bank={}
            for short in ([False,True] if split=='train' and a.arm=='cog_revision' else [False]):
                name='short' if short else 'long'
                stops=[len(all_tokens)] if split=='train' else [len(doc['prefix'])+d for d in (256,1024,4096,16384)]
                previous=0
                state=None
                table=Successors(256)
                hidden=[]
                looked=[]
                queries={}
                for stop in stops:
                    pos=previous
                    while pos<stop:
                        if short and pos%128==0:
                            state=None
                            table=Successors(256)
                        end=min(stop,(pos//128+1)*128) if short else stop
                        hs,state=sequence(m,all_tokens[pos:end],state)
                        hidden.append(hs)
                        looked+=table.apply(all_tokens[pos:end])
                        pos=end
                    qrows=[]
                    for q in doc['questions']:
                        def query_features(reset):
                            prior=None if reset else copy.deepcopy(state)
                            idx=Successors(256) if reset else copy.deepcopy(table)
                            out=[]
                            marks=[]
                            for j,t in enumerate(q['q']):
                                if short and (stop+j)%128==0:
                                    prior=None
                                    idx=Successors(256)
                                h,prior=sequence(m,[t],prior)
                                out.append(h)
                                marks+=idx.apply([t])
                            return torch.cat(out),torch.tensor(marks)
                        qrows.append((query_features(False),query_features(True)))
                    queries[stop]=qrows
                    previous=stop
                bank[name]=(torch.cat(hidden),torch.tensor(looked),queries)
            temp=path.with_suffix('.tmp')
            torch.save(bank,temp)
            temp.replace(path)
        print(a.arm,split,len(docs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parents','out','data'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
