import csv
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'base'))
import study as prior
import run as b

D=json.loads((ROOT/'protocol.json').read_text())
CUES=['accept','reject','accepted','discarded']


def encode(text, keys):
    words=text.split()
    spans=[b.find(text,k) for k in keys]
    positions=[i for i,w in enumerate(words) if w in CUES]
    assert len(positions)==2 and spans[0][0]<spans[1][0]
    cues=[CUES.index(words[i]) for i in positions]
    scores=[]
    for left,right in (positions,positions[::-1]):
        assignment=[]
        for (start,end),pos in zip(spans,(left,right)):
            gap=words[min(end,pos+1):max(start,pos)]
            assignment.append([np.log1p(len(gap)),sum(w in ('but','while','and') for w in gap)])
        scores.append(np.sum(assignment,axis=0))
    # Probability that the earlier cue governs the earlier phrase.
    delta=np.r_[1., scores[0]-scores[1]]
    cue_diff=np.eye(4)[cues[0]]-np.eye(4)[cues[1]]
    local=[prior.feature(words,*s,True) for s in spans]
    return local,delta,cue_diff,cues


def training(seed):
    rng=random.Random(seed)
    for _ in range(D['train_pairs']):
        keys=prior.phrases(rng,b.TRAIN_WORDS,rng.randint(1,3),2)
        frame,names=rng.choice(D['training']),rng.sample(b.WORDS,2)
        for j in (0,1):
            yield encode(prior.render(frame,keys,j,names),keys),j


def fit(seed):
    rows=list(training(seed))
    vec=DictVectorizer()
    x=vec.fit_transform([v for (local,_,_,_),_ in rows for v in local])
    diff=x[::2]-x[1::2]
    y=np.array([j==0 for _,j in rows],float)
    local=LogisticRegression(fit_intercept=False).fit(diff,y)
    a=np.stack([r[1] for r,_ in rows])
    c=np.stack([r[2] for r,_ in rows])
    def loss(w):
        align=expit(a@w[:3])
        positive=expit(c@w[3:])
        p=align*positive+(1-align)*(1-positive)
        return -np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))+.001*(w@w)
    # Distance prior breaks the symmetric attachment/polarity parameterization.
    fit=minimize(loss,[0.,-0.5,0.,0.,0.,0.,0.],method='BFGS')
    assert fit.success,fit.message
    np.savez(ROOT/f'weights-{seed}.npz',wide=local.coef_,features=vec.get_feature_names_out().astype(str),pair=fit.x)
    return dict(zip(vec.get_feature_names_out(),local.coef_[0])),fit.x


def direction(score):
    return -1 if abs(score)<1e-10 else int(score<0)


def evaluate(seed,writer):
    weights,w=fit(seed)
    rng=random.Random(37001)
    supports,_,targets,_=b.data(37002,D['episodes'],'train')
    reader=b.Reader(seed,True)
    ck=np.load(ROOT/'base'/f'recurrent-{seed}.npz')
    reader.E,reader.a,reader.b=(ck[k] for k in ('E','a','b'))
    gate=np.load(ROOT/'base'/f'commit-{seed}.npz')['predicate']
    store=reader.forward(supports)[0].reshape(len(targets),8,2,32).sum(2)
    enc=np.zeros((len(b.WORDS),6),int)
    enc[:,0],enc[:,1]=b.V['entry'],[b.V[k] for k in b.WORDS]
    loc=b.choose(reader.forward(enc)[0]@store.transpose(0,2,1))
    for i,(support,target) in enumerate(zip(supports,targets)):
        names=[next(k for k in b.WORDS if b.V[k] in support[2*t]) for t in target]
        flip=rng.randrange(2)
        keys=prior.phrases(rng,b.TEST_WORDS,1+i%3,2)
        for frame,tmp in D['evaluation'].items():
            for j in (0,1):
                text=prior.render(tmp,keys,j,names)
                features,a,c,cues=encode(text,keys)
                z=[sum(weights.get(k,0)*v for k,v in f.items()) for f in features]
                pair=(2*expit(a@w[:3])-1)*(2*expit(c@w[3:])-1)
                arms=dict(wide=direction(z[0]-z[1]),pair=direction(pair),
                          order=int(cues[0] in (1,3)),oracle=j)
                for arm,pick in arms.items():
                    slot=-1 if pick<0 else int(loc[i,b.WORDS.index(names[pick^flip])])
                    gold=int(target[j^flip])
                    ok,wrong=prior.effects(slot,gold,target,gate)
                    writer.writerow((seed,i,frame,j,text,arm,pick,gold,slot,int(ok),wrong))


if __name__=='__main__':
    with gzip.open(ROOT/'raw.csv.gz','wt') as f:
        writer=csv.writer(f)
        writer.writerow('seed episode frame gold text arm pick goldslot slot effect wrong'.split())
        for seed in D['seeds']:
            evaluate(seed,writer)
