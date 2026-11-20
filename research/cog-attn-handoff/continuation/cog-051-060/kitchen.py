"""Ground real participant narrations in the official EPIC verb/noun action annotations."""
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from scipy.stats import beta
D=Path(os.environ.get('EPIC_DATA_DIR', '/workspace/cog-next-private/epic'))
R=Path(__file__).resolve().parent/'results'
def read(name):
    with (D/f'EPIC_100_{name}.csv').open() as f:return list(csv.DictReader(f))
def h(s):return int(hashlib.sha256(s.encode()).hexdigest(),16)

def fit(x,y):
    def branch(ix):
        counts=Counter(y[ix]); classes=sorted(counts,key=lambda k:(-counts[k],k))
        if len(classes)==1:return int(classes[0])
        groups=[set(),set()]; totals=[0,0]
        for k in classes:
            g=int(totals[1]<totals[0]);groups[g].add(k);totals[g]+=counts[k]
        side=np.isin(y[ix],list(groups[1]))
        model=LogisticRegression(C=20,solver='liblinear',max_iter=150).fit(x[ix],side)
        return model,branch(ix[~side]),branch(ix[side])
    return branch(np.arange(len(y)))
def predict(tree,x):
    answer=np.zeros(x.shape[0],int);p=np.ones(x.shape[0])
    def walk(node,idx):
        if not len(idx):return
        if isinstance(node,int):
            answer[idx]=node;return
        model,left,right=node
        prob=model.predict_proba(x[idx])[:,1];side=prob>.5
        p[idx]*=np.maximum(prob,1-prob)
        walk(left,idx[~side]);walk(right,idx[side])
    walk(tree,np.arange(x.shape[0]));return answer,p
def main():
    all_train=read('train');test=read('validation')
    train=[r for r in all_train if h(r['video_id'])%7];cal=[r for r in all_train if not h(r['video_id'])%7]
    ids=np.random.default_rng(5152).choice(len(train),18000,replace=False);train=[train[i] for i in ids]
    assert {r['video_id'] for r in all_train}.isdisjoint({r['video_id'] for r in test})
    seenpeople={r['participant_id'] for r in train}
    lex={t for r in train for k in ('verb','noun') for t in re.findall(r'[a-z]+',r[k].lower())}
    def mask(s):return re.sub(r'[a-z]+',lambda m:' xx ' if m[0] in lex else m[0],s.lower())
    text=[r['narration'] for r in train+cal+test]+[mask(r['narration']) for r in test]
    vectors=[TfidfVectorizer(ngram_range=(1,2),max_features=9000,min_df=2),
             TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=13000,min_df=3)]
    x=hstack([v.fit(text[:len(train)]).transform(text) for v in vectors],format='csr')
    labels=np.array([[int(r['verb_class']),int(r['noun_class'])] for r in train+cal+test+test])
    pair=labels[:,0]*10000+labels[:,1]
    assert np.max(labels[:,1])<10000
    n=len(train);z=len(cal);e=len(test)
    trained=[];pred=[]
    for target in [labels[:,0],labels[:,1],pair]:
        tree=fit(x[:n],target[:n]);trained.append(tree)
        pred.append(predict(tree,x[n:]))
    joblib.dump((vectors,trained),D/'models.joblib',compress=3)
    fresh=~np.isin(pair[n+z:n+z+e],pair[:n])
    known=np.isin(labels[n+z:n+z+e,0],labels[:n,0])&np.isin(labels[n+z:n+z+e,1],labels[:n,1])
    unseen=np.array([r['participant_id'] not in seenpeople for r in test])
    res={}
    for name,(a,p) in {'factorized':(pred[0][0]*10000+pred[1][0],pred[0][1]*pred[1][1]),'flat':pred[2]}.items():
        correct=a==pair[n:]
        options=[t for t in np.arange(.01,1.0,.01) if (p[:z]>=t).any() and correct[:z][p[:z]>=t].mean()>=.99]
        threshold=min(options,default=2)
        res[name]={'cutoff':float(threshold)}
        for split,idx in [('all',np.arange(e)),('new_pair',np.flatnonzero(fresh&known)),('unseen_people',np.flatnonzero(unseen)),('masked',np.arange(e)+e)]:
            ok=correct[z+idx];admit=p[z+idx]>=threshold
            res[name][split]=dict(n=len(idx),correct=int(ok.sum()),admitted=int(admit.sum()),tp=int(ok[admit].sum()))
        # Public per-observation evidence carries IDs, annotations and numeric predictions only.
        res[name]['rows']=[[r['narration_id'],int(pair[n+z+i]),int(a[z+i]),float(p[z+i])] for i,r in enumerate(test)]
    (R/'kitchen.json').write_text(json.dumps(dict(train=n,cal=z,validation=e,source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*.csv')},arms=res)))
if __name__=='__main__':main()
