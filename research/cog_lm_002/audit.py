"""Validate the published parents, exposure schedule, saved fit identity and replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from adapt import digest
from data import corpus,train_offsets
from runtime import Runtime


def run(a):
    meta=json.loads((Path(__file__).resolve().parents[1]/'cog_lm_001/evidence/metrics.json').read_text())
    parent=a.original/'scale-local-17-continued.pt'
    short=a.original/'scale-mamba2-17.pt'
    continued=a.raw/'scale-mamba2-17-continued.pt'
    for p in (parent,short):
        assert digest(p)==meta[p.stem]['training']['checkpoint_sha256']
    gap=json.loads(continued.with_suffix('.train.json').read_text())
    old=meta[parent.stem]['training']
    assert gap['offsets_sha256']==old['offsets_sha256'] and gap['total_tokens']==old['total_tokens']==524288
    assert gap['parent_sha256']==digest(short) and gap['checkpoint_sha256']==digest(continued)
    train,bounds=corpus(a.data,'train')
    exposure=np.zeros(len(train),dtype=bool)
    for seed,steps in ((17,128),(1017,896)):
        _,offs=train_offsets(a.data,seed,steps)
        exposure[(offs.reshape(-1,1)+np.arange(1,129)).ravel()]=True
    first=int(exposure.sum())
    starts=np.concatenate([np.arange(lo,hi-128,128,dtype=np.int64) for lo,hi in zip(bounds,bounds[1:])])
    order=np.random.default_rng(20261121).choice(starts,12288,replace=False)
    sha=hashlib.sha256(order.tobytes()).hexdigest()
    targets=(order[:,None]+np.arange(1,129)).ravel()
    assert len(np.unique(targets))==len(targets)==1572864
    exposure[targets]=True
    tokenizer=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    test,offsets=corpus(a.data,'test')
    x=torch.tensor(test[offsets[0]+48:offsets[0]+145].astype('int64'))[None]
    result=dict(distinct_parent_positions=first,distinct_total_positions=int(exposure.sum()),
        corpus_positions=len(train),additional_target_positions=len(targets),offsets_sha256=sha,replay={})
    for joint in (False,True):
        for arm in ('mamba_lookup','cog_lookup','cog_revision'):
            root=a.joint if joint else a.raw
            name=('joint_' if joint else '')+arm
            f=json.loads((root/(name+'.json')).read_text())
            p=continued if arm=='mamba_lookup' else parent
            assert digest(p)==f['parent_sha256'] and f['offsets_sha256']==sha
            base=root/(name+'.base.pt') if joint else p
            head=root/(name+'.pt')
            assert digest(head)==f['adapter_sha256']
            if joint:
                assert digest(base)==f['base_sha256']
            m=Runtime.load(base,head)
            native,_=m.advance(x)
            _,state=m.advance(x[:,:32])
            replay,_=m.advance(x[:,32:],copy.deepcopy(state))
            err=float((native[:,32:]-replay).abs().max())
            assert err < .00012,(name,err)
            samples=[]
            for prompt in ('The history of the city','When the rain stopped,','The customer asked'):
                generator=torch.Generator().manual_seed(31)
                logp,stream=m.advance(torch.tensor([tokenizer.encode(prompt).ids]),last_only=True)
                ids=[]
                for _ in range(48):
                    nxt=torch.multinomial((logp[:,-1]/.8).softmax(-1),1,generator=generator)
                    ids.append(nxt.item())
                    if nxt.item()==0:
                        break
                    logp,stream=m.advance(nxt,stream,last_only=True)
                samples.append(dict(prompt=prompt,ids=ids,completion=tokenizer.decode(ids)))
            result['replay'][name]=dict(max_abs_logp=err,generated=m.generate(x,16),samples=samples)
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('raw','joint','original','data','out'):
        parser.add_argument('--'+k,type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(2)
    run(a)
