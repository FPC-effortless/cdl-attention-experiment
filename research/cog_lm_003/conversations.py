"""Authentic revisions on known and fresh shards, sharing exact frozen backbone computation."""
import argparse
from common import *
from fit import CONFIGS
from dialogue import cases,mentioned


@torch.inference_mode()
def evaluate(a):
    torch.set_num_threads(a.threads)
    base,prior=model_paths(a.parents,a.arm)
    system=Runtime.load(base,prior)
    variants={name:Runtime.load(base,a.out/(name+'.pt')).head for name in CONFIGS if CONFIGS[name][0]==a.arm}
    variants['parent_'+a.arm]=system.head
    if a.arm=='cog_revision':
        variants['route_revision_off']=variants['route_revision']
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    for shard,path in [('known',a.known),('fresh',a.fresh)]:
        caseset=list(cases(path))
        records={k:[dict(info) for info,_,_ in caseset] for k in variants}
        tasks=[]
        for idx,(info,before,after) in enumerate(caseset):
            for condition,prompt in (('before',before),('after',after)):
                n=len(tok.encode(prompt).ids)
                raw=prompt.split('\nWhat ')[0]
                last={v:max((m.start() for m in mentioned(info[v],raw)),default=-1) for v in ('old','new')}
                for name in records:
                    records[name][idx][condition+'_literal']=last['new']>last['old']
                for answer in ('old','new'):
                    ids=tok.encode(prompt+' '+info[answer]+'.').ids
                    assert ids[:n]==tok.encode(prompt).ids
                    tasks.append((idx,condition,answer,n,ids))
        tasks.sort(key=lambda x:len(x[-1]))
        for i in range(0,len(tasks),4):
            batch=tasks[i:i+4]
            T=16*((max(len(t[-1])-1 for t in batch)+15)//16)
            x=torch.zeros(len(batch),T,dtype=torch.long)
            for j,(_,_,_,_,ids) in enumerate(batch):
                x[j,:len(ids)-1]=torch.tensor(ids[:-1])
            states=None
            hh=[]
            for chunk in x.split(256,1):
                h,states=features(system.model,chunk,states)
                hh.append(h)
            h=torch.cat(hh,1)
            index=successor_ids(x,256)
            for name,head in variants.items():
                pred,_=head(h,index,system.model.emb.weight,mask_read=name.endswith('_off'))
                for j,(idx,condition,answer,n,ids) in enumerate(batch):
                    score=pred[j,n-1:len(ids)-1]
                    gold=torch.tensor(ids[n:])
                    row=records[name][idx]
                    row[condition+'_'+answer]=float(score.gather(-1,gold[:,None]).sum())
                    row[condition+'_'+answer+'_exact']=bool((score.argmax(-1)==gold).all())
        for name,rows in records.items():
            save(a.out/(name+'.'+shard+'.json'),rows)
        print(a.arm,shard,len(caseset),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parents','out','data','known','fresh'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--threads',type=int,default=2)
    evaluate(p.parse_args())
