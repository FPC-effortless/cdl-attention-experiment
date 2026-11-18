import csv
import gzip
import json
import random
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import run as b
from roles import State, outcome

ROOT = Path(__file__).parent
D = json.loads((ROOT/'design.json').read_text())
PERMS = list(permutations(range(3)))


def render(frame, keys, winner, names):
    x,y = keys
    u,v = ('accept','reject')[::1 if winner==0 else -1]
    m,n = ('accepted','discarded')[::1 if winner==0 else -1]
    return 'for %s and %s ' % tuple(names) + frame.format(x=x,y=y,u=u,v=v,m=m,n=n,p=keys[winner],d=keys[1-winner])


def feature(words, a, z, wide):
    result = b.features(words,a,z)
    if wide:
        for distance in range(1,7):
            for side, pos in [('L',a-distance),('R',z+distance-1)]:
                token = words[pos] if 0<=pos<len(words) else '<edge>'
                result[f'{side}{distance}=' + ('<record>' if token in b.WORDS else token)] = 1
    return result


def phrases(rng, vocab, size, count):
    words = rng.sample(vocab, size*count)
    return [' '.join(words[j:j+size]) for j in range(0,len(words),size)]


def fit(seed,wide):
    rng = random.Random(seed)
    features, labels = [], []
    for _ in range(D['train_pairs']):
        keys = phrases(rng,b.TRAIN_WORDS,rng.randint(1,3),2)
        frame,names = rng.choice(D['train']),rng.sample(b.WORDS,2)
        for winner in (0,1):
            text = render(frame,keys,winner,names)
            target = b.find(text,keys[winner])
            words = text.split()
            for a,z in b.spans(text):
                features.append(feature(words,a,z,wide))
                labels.append((a,z)==target)
    vec = DictVectorizer()
    clf = LogisticRegression(max_iter=300,class_weight='balanced').fit(vec.fit_transform(features),labels)
    np.savez(ROOT/f'fit-{seed}-{int(wide)}.npz',names=vec.get_feature_names_out().astype(str),coef=clf.coef_,intercept=clf.intercept_)
    weights = dict(zip(vec.get_feature_names_out(),clf.coef_[0]))
    @lru_cache(maxsize=8192)
    def select(text):
        words=text.split()
        candidates=b.spans(text)
        logits=np.array([sum(weights.get(k,0)*v for k,v in feature(words,a,z,wide).items()) for a,z in candidates])
        idx=int(b.choose(logits))
        return b.key_for(text,candidates[idx]) if idx>=0 else None
    return select


def effects(slot,gold,target,gate):
    verified=True
    wrong=0
    for present in target:
        cells=[None]*10
        cells[present],cells[8],cells[9]=('v',4),('v',7),('v',3)
        for op in ('copy','link'):
            actual=outcome(State(tuple(cells)),op,slot,gate)
            allow=present==gold
            expected=cells.copy()
            expected[8]=('v',91)
            if allow:
                expected[9]=('v',7) if op=='copy' else ('l',8)
            verified &= actual==(allow,tuple(expected),7 if allow and op=='copy' else 91 if allow else 3)
            wrong+=int(actual[0] and not allow)
    return verified,wrong


def evaluate(seed,mode,select,writer):
    rng=random.Random(36101)
    supports,_,targets,_=b.data(36102,D['episodes'],'train')
    reader=b.Reader(seed,True)
    ck=np.load(ROOT/f'recurrent-{seed}.npz')
    reader.E,reader.a,reader.b=(ck[k] for k in ('E','a','b'))
    gate=np.load(ROOT/f'commit-{seed}.npz')['predicate']
    enc=np.zeros((len(b.WORDS),6),int)
    enc[:,0],enc[:,1]=b.V['entry'],[b.V[w] for w in b.WORDS]
    stores=reader.forward(supports)[0].reshape(len(supports),8,2,32).sum(2)
    addresses=b.choose(reader.forward(enc)[0]@stores.transpose(0,2,1))
    for i,(support,target) in enumerate(zip(supports,targets)):
        names=[next(w for w in b.WORDS if b.V[w] in support[2*j]) for j in range(8)]
        other=[w for j,w in enumerate(names) if j not in target]
        keys=phrases(rng,b.TEST_WORDS,i%3+1,3)
        flip=rng.randrange(2)
        directions=[flip,1-flip,rng.randrange(2)]
        foil=[rng.choice([k for k in range(3) if k!=j]) for j in range(3)]
        positions=[rng.randrange(2) for _ in keys]
        for perm in PERMS:
            bindings=defaultdict(set)
            for j,frameidx in enumerate(perm):
                pair=[keys[j],keys[foil[j]]][::1 if positions[j]==0 else -1]
                mentions=other[2*j:2*j+2]
                text=render(D['train'][frameidx],pair,positions[j],mentions)
                output=mentions[directions[j]]
                picked=keys[j] if mode=='oracle' else select(text)
                if picked:
                    bindings[picked].add(mentions.index(output))
            for layout,frame in D['tests'].items():
                for j in (0,1):
                    mentions=[names[k] for k in target]
                    text=render(frame,keys[:2],j,mentions)
                    picked=keys[j] if mode=='oracle' else select(text)
                    options=bindings.get(picked,set())
                    slot=-1 if len(options)!=1 else int(addresses[i,b.WORDS.index(mentions[next(iter(options))])])
                    gold=int(target[directions[j]])
                    ok,wrong=effects(slot,gold,target,gate)
                    writer.writerow([seed,mode,i,''.join(map(str,perm)),layout,j,text,keys[j],picked,gold,slot,int(ok),wrong])


if __name__=='__main__':
    with gzip.open(ROOT/'raw.csv.gz','wt') as handle:
        writer=csv.writer(handle)
        writer.writerow('seed mode episode perm layout query text gold_key key gold_slot slot effects wrong_commits'.split())
        for seed in D['seeds']:
            for mode in ('local','wide','oracle'):
                evaluate(seed,mode,None if mode=='oracle' else fit(seed,mode=='wide'),writer)
