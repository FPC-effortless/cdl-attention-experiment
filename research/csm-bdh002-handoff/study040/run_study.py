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
                  q=references[order], t=references[1-order], m=('accepted','discarded')[approval], n=('discarded','accepted')[approval])
    values['r' if approval==0 else 's'] += '*'
    words = ('of '+' and '.join(phrases)+' '+template.format(**values)).split()
    marks = [i for i,w in enumerate(words) if w.endswith('*')]
    (target,) = marks
    words[target] = words[target][:-1]
    return words, target


def argument_features(words, ref, cue, lo, hi):
    between = words[lo+1:hi]
    return {
        'cue_to_argument': int(cue<ref and words[ref-2:ref]==['to','the']),
        'argument_receives_cue': int(ref<cue and words[ref+1:ref+2]==['receives']),
        'contrast_marker': int('rather' in between),
        'same_segment': int(not any(w in boundaries for w in between)),
    }


def candidates(words, structured=False):
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
            features = dict(gap=np.log1p(hi-lo),cross=abs(clauses[ref]-clauses[cue]),
                            after=int(ref>cue),**{'near='+w:1 for w in local})
            if structured:
                features.update(argument_features(words,ref,cue,lo,hi))
            yield ref,cue,features


def fit(seed, structured):
    rng = random.Random(seed)
    features,labels = [],[]
    for _ in range(protocol['train_quartets']):
        phrases = prior.phrases(rng,base.TRAIN_WORDS,rng.randint(1,3),2)
        template = rng.choice(protocol['train'])
        for order in range(2):
            for approval in range(2):
                words,target = fixture(template,order,approval,phrases)
                for ref,_,row in candidates(words,structured):
                    features.append(row)
                    labels.append(ref==target)
    vectorizer = DictVectorizer()
    model = LogisticRegression(class_weight='balanced').fit(vectorizer.fit_transform(features),labels)
    weights = dict(zip(vectorizer.get_feature_names_out(),model.coef_[0]))
    (ROOT/f'weights-{seed}-{int(structured)}.json').write_text(json.dumps(weights,indent=2))
    return weights


def predict(rows,weights):
    scores = {}
    for ref,_,features in rows:
        score = sum(weights.get(k,0.)*v for k,v in features.items())
        scores[ref] = max(score,scores.get(ref,-np.inf))
    picked = int(base.choose(np.array(list(scores.values()))))
    return list(scores)[picked] if picked>=0 else -1


def evaluate(seed, writer, extra_selector=None):
    plain, structured = fit(seed,False), fit(seed,True)
    rng = random.Random(40001)
    supports,_,targets,_ = base.data(40002,protocol['test_quartets'],'train')
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
                    paths = list(candidates(words,True))
                    clause = min(rows,key=lambda r:(r[2]['cross'],r[2]['gap']))[0]
                    rule = min(paths,key=lambda r:(r[2]['cross'],
                               -max(r[2]['cue_to_argument'],r[2]['argument_receives_cue']),
                               r[2]['gap']))[0]
                    goldslot = int(slots[references.index(words[gold])^flip])
                    predictions = dict(clause=clause, plain=predict(rows,plain),
                                       structured=predict(paths,structured), rule=rule, oracle=gold)
                    if extra_selector is not None:
                        predictions['scope_grammar'] = extra_selector(words)
                    for arm,ref in predictions.items():
                        slot = -1 if ref<0 else int(addresses[episode,base.WORDS.index(names[references.index(words[ref])^flip])])
                        exact,wrong = prior.effects(slot,goldslot,slots,gate)
                        writer.writerow((seed,episode,layout,order,approval,' '.join(words),arm,gold,ref,goldslot,slot,int(exact),wrong))


if __name__ == '__main__':
    with gzip.open(ROOT/'raw.csv.gz','wt') as f:
        writer = csv.writer(f)
        writer.writerow('seed episode layout reforder approval text arm gold ref goldslot slot effect wrong'.split())
        for seed in protocol['seeds']:
            evaluate(seed,writer)
