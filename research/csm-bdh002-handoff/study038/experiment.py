import csv
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'base'))
import study as prior
import run as b

D = json.loads((ROOT / 'protocol.json').read_text())
TRAIN_WORDS = [('former', 'latter'), ('first', 'second')]
ALIAS = ('earlier', 'later')
INVENTORY = dict(word for pair in [*TRAIN_WORDS, ALIAS] for word in zip(pair, (0, 1)))
PHRASE_WORDS = b.TEST_WORDS


def prompt(frame, keys, references, order, approval):
    return frame.format(x=keys[0], y=keys[1], r=references[order], s=references[1-order],
                        m=('accepted','discarded')[approval], n=('discarded','accepted')[approval])


def approval_reference(text):
    words = text.split()
    cues = [i for i,w in enumerate(words) if w in ('accepted','discarded')]
    assert len(cues) == 2
    refs = [i for i,w in enumerate(words) if w in INVENTORY]
    assert len(refs) >= 2
    # Find two distinct overt references. Favor nearby references and break
    # equal costs by preserving textual order between statuses and references.
    _, _, first, second = min(
        (abs(cues[0]-i)+abs(cues[1]-j), i>j, i,j)
        for i in refs for j in refs if i!=j
    )
    return words[first if words[cues[0]]=='accepted' else second]


def fit(seed):
    rng = random.Random(seed)
    features, labels = [], []
    for _ in range(D['train_quartets']):
        keys = prior.phrases(rng, b.TRAIN_WORDS, rng.randint(1,3), 2)
        frame, refs = rng.choice(D['train']), rng.choice(TRAIN_WORDS)
        for order in range(2):
            for approval in range(2):
                text = prompt(frame, keys, refs, order, approval)
                features.append({approval_reference(text): 1})
                labels.append(order^approval)
    vec = DictVectorizer()
    model = LogisticRegression(fit_intercept=False).fit(vec.fit_transform(features),labels)
    weights = dict(zip(vec.get_feature_names_out().tolist(),model.coef_[0].tolist()))
    (ROOT/f'weights-{seed}.json').write_text(json.dumps(weights,indent=2))
    return weights


def evaluate(seed, writer):
    weights = fit(seed)
    rng = random.Random(38001)
    supports,_,targets,_ = b.data(38002,D['eval_quartets'],'train')
    reader = b.Reader(seed,True)
    ck = np.load(ROOT/'base'/f'recurrent-{seed}.npz')
    reader.E,reader.a,reader.b = (ck[k] for k in ('E','a','b'))
    gate = np.load(ROOT/'base'/f'commit-{seed}.npz')['predicate']
    enc = np.zeros((len(b.WORDS),6),int)
    enc[:,0],enc[:,1] = b.V['entry'],[b.V[k] for k in b.WORDS]
    bank = reader.forward(supports)[0].reshape(len(targets),8,2,32).sum(2)
    addresses = b.choose(reader.forward(enc)[0]@bank.transpose(0,2,1))
    for i,(support,target) in enumerate(zip(supports,targets)):
        names = [next(k for k in b.WORDS if b.V[k] in support[2*t]) for t in target]
        flip = rng.randrange(2)
        keys = prior.phrases(rng,PHRASE_WORDS,1+i%3,2)
        for frame,template in D['tests'].items():
            refs = ALIAS if frame=='unseen_alias' else TRAIN_WORDS[i%2]
            for order in range(2):
                for approval in range(2):
                    text = prompt(template,keys,refs,order,approval)
                    reference = approval_reference(text)
                    gold = order^approval
                    weight = weights.get(reference,0.)
                    picks = dict(learned=int(weight>0) if weight else -1,
                                 dictionary=INVENTORY[reference] if reference in weights else -1,
                                 expanded=INVENTORY[reference],order=approval,oracle=gold)
                    goldslot = int(target[gold^flip])
                    for arm,pick in picks.items():
                        slot = -1 if pick<0 else int(addresses[i,b.WORDS.index(names[pick^flip])])
                        exact,wrong = prior.effects(slot,goldslot,target,gate)
                        writer.writerow((seed,i,frame,order,approval,text,reference,arm,gold,pick,goldslot,slot,int(exact),wrong))


if __name__=='__main__':
    with gzip.open(ROOT/'raw.csv.gz','wt') as handle:
        writer=csv.writer(handle)
        writer.writerow('seed episode frame reforder approval text reference arm gold pick goldslot slot effect wrong'.split())
        for seed in D['seeds']:
            evaluate(seed,writer)
