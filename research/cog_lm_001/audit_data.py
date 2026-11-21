"""Check pristine source statistics, sampled target overlap and training coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
from data import corpus, eval_offsets, train_offsets


def verify(root):
    train,bounds=corpus(root,'train')
    haystack=train.tobytes()
    summary={}
    for split in ('validation','test','transfer'):
        tokens,rows=eval_offsets(root,split)
        matches=[]
        for i,offset in rows:
            query=tokens[offset:offset+129].tobytes()
            start=haystack.find(query)
            while start!=-1:
                t=start//2
                d=int(np.searchsorted(bounds,t,side='right')-1)
                if start%2==0 and t+129<=bounds[d+1]:
                    matches.append([i,offset,int(t)])
                    break
                start=haystack.find(query,start+1)
        summary[split]=dict(documents=len({i for i,_ in rows}),windows=len(rows),targets=128*len(rows),
                            exact_train_window_matches=matches)
    coverage={}
    for name,offsets in [('compact-'+str(s),train_offsets(root,s,256)[1]) for s in (17,29,43)]+[
            ('scale-17',train_offsets(root,17,128)[1]),('scale-extended-17',np.concatenate(
                [train_offsets(root,17,128)[1],train_offsets(root,1017,896)[1]]))]:
        observed=np.zeros(len(train),dtype=bool)
        for t in offsets.flatten():
            observed[t+1:t+129]=True
        coverage[name]=dict(targets_drawn=offsets.size*128,distinct_corpus_target_positions=int(observed.sum()),
                            documents_sampled=len(np.unique(np.searchsorted(bounds,offsets,side='right')-1)))
    return dict(panels=summary,training=coverage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.write_text(json.dumps(verify(a.data),indent=2)+'\n')
