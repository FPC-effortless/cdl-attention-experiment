"""Exchange read-time bank states between documents, retaining original queries and writes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from data import eval_offsets
from model import restore


@torch.inference_mode()
def score(model,data,split):
    tokens,rows=eval_offsets(data,split)
    books=defaultdict(list)
    for d,o in rows:
        books[d].append(o)
    ordered=[(d,v[j]) for j in (0,1) for d,v in books.items() if len(v)>j]
    out=[]
    for start in range(0,len(ordered),4):
        batch=ordered[start:start+4]
        if len(batch)<2 or len({d for d,_ in batch})!=len(batch):
            raise ValueError('donor must be another document')
        seq=torch.tensor(np.array(tokens[np.array([o for _,o in batch])[:,None]+np.arange(129)]),dtype=torch.long)
        vals=[]
        for mode in ('full','foreign_slots'):
            p,_=model(seq[:,:-1],intervention=mode)
            vals.append(-p.gather(-1,seq[:,1:,None]).squeeze(-1).sum(1))
        out.extend(dict(doc=d,offset=o,donor=batch[j-1][0],count=128,intact=vals[0][j].item(),
                        foreign=vals[1][j].item()) for j,(d,o) in enumerate(batch))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);m=restore(a.weights)
    result={s:score(m,a.data,s) for s in ('test','transfer')}
    (a.out/(a.weights.stem+'.foreign.json')).write_text(json.dumps(result)+'\n')
