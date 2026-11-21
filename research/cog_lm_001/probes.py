"""Counterbalanced, zero-shot memory interventions. Gold is evaluator-only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from evaluate import restore
from model import Successors
from data import corpus


def trials(data):
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'))
    tokens,off=corpus(data,'validation')
    encode=lambda text:tokenizer.encode(text).ids
    for distance in (64,256,1024):
        for kind in ('repeat','revision','reword'):
            for i,key in enumerate('ABCDEFGH'):
                choices=('north','south') if i%2==0 else ('east','west')
                # Natural intervening prose. Each orientation uses the identical filler.
                start=next(int(lo) for lo,hi in zip(off[i:],off[i+1:]) if hi-lo>distance+60)
                filler=list(map(int,tokens[start+45:start+45+distance]))
                clause=' The current direction for station '+key+' is'
                for flip in (0,1):
                    old,new=choices[flip],choices[1-flip]
                    answer=encode(' '+new)
                    stale=encode(' '+old)
                    assert len(answer)==len(stale)==1
                    prefix=encode(clause+' '+new+'.')
                    if kind!='repeat':
                        prefix=encode(clause+' '+old+'. Correction.'+clause+' '+new+'.')
                    question=clause if kind!='reword' else ' The direction reported by station '+key+' points to'
                    sequence=prefix+filler+encode(question)
                    found=Successors(256).apply(sequence)[-1]
                    yield dict(kind=kind,distance=distance,key=key,flip=flip),sequence,answer[0],stale[0],found


@torch.no_grad()
def run(args):
    m=restore(args.weights)
    rows=[]
    for identity,seq,new,old,found in trials(args.data):
        x=torch.tensor([seq])
        z,_=m(x)
        pred=z[0,-1]
        rows.append(dict(**identity,tokens=len(seq),target=new,old=old,
                         top=int(pred.argmax()),preferred=bool(pred[new]>pred[old]),
                         target_logp=float(pred[new]),old_logp=float(pred[old]),index_prediction=found))
    (args.out/(args.weights.stem+'.probe.json')).write_text(json.dumps(rows)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);run(a)
