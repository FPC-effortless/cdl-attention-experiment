"""Execute every learned bank operation while suppressing its contribution to output."""
import argparse
import json
from pathlib import Path
import time

import torch
from runtime import Runtime
from data import eval_offsets


def run(a):
    m = Runtime.load(a.base,a.adapter)
    if m.head.bank is None:
        raise ValueError('requires a revision bank')
    systems = dict(full=m,masked=Runtime(m.model,m.head,mask_read=True),disabled=Runtime(m.model,m.head,disabled=True))
    tokens, rows = eval_offsets(a.data,'test')
    results = []
    for i,(article,start) in enumerate(rows):
        raw = torch.tensor(tokens[start:start+129].astype('int64'))[None]
        record = dict(article=article,offset=start)
        outputs = {}
        names = list(systems)
        for name in names[i%3:] + names[:i%3]:
            before = time.perf_counter()
            pred,state = systems[name].advance(raw[:,:-1])
            record[name+'_s'] = time.perf_counter()-before
            outputs[name]=pred
            record[name+'_nll'] = float(-pred.gather(-1,raw[:,1:,None]).sum())
        record['masked_disabled_max_delta'] = float((outputs['masked']-outputs['disabled']).abs().max())
        results.append(record)
    (a.out/(a.adapter.stem+'.matched.json')).write_text(json.dumps(results)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','adapter','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    run(args)
