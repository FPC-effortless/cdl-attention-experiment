import csv
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'base'))
import study as prior
import run as base

protocol = json.loads((ROOT/'protocol.json').read_text())
references = ('former','latter')
boundaries = {';','.','!'}


def fixture(template, order, approval, phrases):
    # Mark the target occurrence before rendering. Remove the mark before scoring.
    values = dict(r=references[order], s=references[1-order], d=references[1-order^approval],
                  m=('accepted','discarded')[approval], n=('discarded','accepted')[approval])
    values['r' if approval==0 else 's'] += '*'
    words = ('of '+' and '.join(phrases)+' '+template.format(**values)).split()
    target = next(i for i,w in enumerate(words) if w.endswith('*'))
    words[target] = words[target][:-1]
    return words, target


def candidates(words):
    clauses, clause = [], 0
    for word in words:
        clauses.append(clause)
        clause += word in boundaries
    for ref,word in enumerate(words):
        if word not in references:
            continue
        for cue,status in enumerate(words):
            if status != 'accepted':
                continue
            lo,hi = sorted((ref,cue))
            local = words[max(0,ref-2):ref] + words[ref+1:ref+3]
            yield ref,cue,dict(gap=np.log1p(hi-lo),cross=abs(clauses[ref]-clauses[cue]),
                                after=int(ref>cue),**{'near='+w:1 for w in local})


def fit(seed):
    rng = random.Random(seed)
    features,labels = [],[]
    for _ in range(protocol['train_quartets']):
        phrases = prior.phrases(rng,base.TRAIN_WORDS,rng.randint(1,3),2)
        template = rng.choice(protocol['train'])
        for order in range(2):
            for approval in range(2):
                words,target = fixture(template,order,approval,phrases)
                for ref,_,row in candidates(words):
                    features.append(row)
                    labels.append(ref==target)
    vectorizer = DictVectorizer()
    model = LogisticRegression(class_weight='balanced').fit(vectorizer.fit_transform(features),labels)
    weights = dict(zip(vectorizer.get_feature_names_out(),model.coef_[0]))
    (ROOT/f'weights-{seed}.json').write_text(json.dumps(weights,indent=2))
    return weights


def evaluate(seed, writer):
    weights = fit(seed)
    rng = random.Random(39001)
    supports,_,targets,_ = base.data(39002,protocol['test_quartets'],'train')
    reader = base.Reader(seed,True)
    with np.load(ROOT/'base'/f'recurrent-{seed}.npz') as ck:
        reader.E,reader.a,reader.b = (ck[k] for k in ('E','a','b'))
    gate = np.load(ROOT/'base'/f'commit-{seed}.npz')['predicate']
    enc = np.zeros((len(base.WORDS),6),int)
    enc[:,0],enc[:,1] = base.V['entry'],[base.V[k] for k in base.WORDS]
    memory = reader.forward(supports)[0].reshape(len(targets),8,2,32).sum(2)
    addresses = base.choose(reader.forward(enc)[0]@memory.transpose(0,2,1))
    for episode,(support,slots) in enumerate(zip(supports,targets)):
        names = [next(k for k in base.WORDS if base.V[k] in support[2*t]) for t in slots]
        flip = rng.randrange(2)
        phrases = prior.phrases(rng,base.TEST_WORDS,1+episode%3,2)
        for layout,template in protocol['test'].items():
            for order in range(2):
                for approval in range(2):
                    words,gold = fixture(template,order,approval,phrases)
                    rows = list(candidates(words))
                    nearest = min(rows,key=lambda r:r[2]['gap'])[0]
                    clause = min(rows,key=lambda r:(r[2]['cross'],r[2]['gap']))[0]
                    learned_scores = {}
                    for ref,_,features in rows:
                        score = sum(weights.get(k,0.)*v for k,v in features.items())
                        learned_scores[ref] = max(score,learned_scores.get(ref,-np.inf))
                    picked = int(base.choose(np.array(list(learned_scores.values()))))
                    learned = list(learned_scores)[picked] if picked>=0 else -1
                    goldslot = int(slots[references.index(words[gold])^flip])
                    for arm,ref in dict(nearest=nearest,clause=clause,learned=learned,oracle=gold).items():
                        slot = -1 if ref<0 else int(addresses[episode,base.WORDS.index(names[references.index(words[ref])^flip])])
                        exact,wrong = prior.effects(slot,goldslot,slots,gate)
                        writer.writerow((seed,episode,layout,order,approval,' '.join(words),arm,gold,ref,goldslot,slot,int(exact),wrong))


if __name__ == '__main__':
    with gzip.open(ROOT/'raw.csv.gz','wt') as f:
        writer = csv.writer(f)
        writer.writerow('seed episode layout reforder approval text arm gold ref goldslot slot effect wrong'.split())
        for seed in protocol['seeds']:
            evaluate(seed,writer)
