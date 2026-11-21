"""Re-run the two decisive literal-delay conditions from 002, intervening only on retrieval."""
import argparse
import copy
from common import Path, np, torch, Tokenizer, Runtime, corpus, save, model_paths, sequence
from panels import CASES


@torch.inference_mode()
def run(a):
    base,head=model_paths(a.parents,'cog_revision')
    m=Runtime.load(base,head)
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    arr,offs=corpus(a.data,'test')
    docs=np.flatnonzero(np.diff(offs)>16384+128)
    rows=[]
    for i,(first,second,av,bv,literal,reword) in enumerate(CASES):
        pos=offs[docs[i]]+64
        for flip in (0,1):
            old,new=(av,bv) if flip==0 else (bv,av)
            prefix=' '.join((literal.format(first)+' '+old+'.',literal.format(second)+' '+new+'.',
                             'Correction.',literal.format(first)+' '+new+'.',literal.format(second)+' '+old+'.'))
            _,state=m.advance(torch.tensor([tok.encode(prefix).ids]),last_only=True)
            seen=0
            for d in (256,1024):
                _,state=m.advance(torch.tensor([arr[pos+seen:pos+d].tolist()]),state,last_only=True)
                seen=d
                for who,answer in ((first,new),(second,old)):
                    q=tok.encode(' '+literal.format(who)).ids
                    layers,bank,table=copy.deepcopy(state)
                    h,_=sequence(m.model,q,layers)
                    index=torch.tensor(table[0].apply(q))[None]
                    active,_=m.head(h[None],index,m.model.emb.weight,bank,last_only=True)
                    erased,_=m.head(h[None],torch.full_like(index,-1),m.model.emb.weight,bank,last_only=True)
                    gold,=tok.encode(' '+answer).ids
                    rows.append(dict(arm='002_joint_cog_revision',family=i,flip=flip,entity=who,delay=d,
                        gold=gold,table=int(index[0,-1]),native=int(active.argmax()),without_lookup=int(erased.argmax())))
    save(a.out/'attribution_002.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    run(a)
