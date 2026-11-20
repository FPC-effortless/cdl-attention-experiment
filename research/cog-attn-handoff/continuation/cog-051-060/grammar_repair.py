"""Few-shot polarity learning with novel clauses and changed clause order."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from panel import Panel,probe
HERE=Path(__file__).resolve().parent

def run(seed):
    rng=np.random.default_rng(seed+100000)
    env=Panel(rng);m=probe(env,rng,True)
    old=['dax','wug','fep','blick','zorp'];new=['zavi','moro','plex','binu','tevo']
    # Neutral single-entity trials teach aliases using confirmed frame changes.
    lex={}
    for wordset in (old,new):
        for a,word in enumerate(wordset):
            while True:
                s=int(rng.integers(32));label=m.label(env.frame(s),env.frame(env.step(s,a)))
                if label is not None: lex[word]=label;break
    phrases=[['must stay steady','should remain as before','keeps its appearance'],
             ['must change appearance','should switch state','needs to change']]
    unseen=[['should stay as before','must remain steady'],['should change state','must switch appearance']]
    unknown=[['is already fine','is settled'],['needs attention','is the exception']]
    docs=[];labels=[]
    for a in range(5):
        for signal in (0,1):
            for wording in phrases[signal]:
                # A positive demonstration may be initially blocked. Retry only on a freshly reset state.
                s=int(rng.integers(32))
                if signal:
                    while env.step(s,a)==s:s=int(rng.integers(32))
                after=env.step(s,a) if signal else s
                docs.append(wording);labels.append(int(np.any(env.frame(after)!=env.frame(s))))
    vec=TfidfVectorizer(ngram_range=(1,2));clf=LogisticRegression(C=10).fit(vec.fit_transform(docs),labels)
    results={}
    for name,forms in [('new_combinations',unseen),('unrelated_wording',unknown)]:
        counts=[0,0,0]
        for _ in range(300):
            a,b=rng.choice(5,2,replace=False)
            # Narrated clauses have explicit coordination; each contains one alias.
            clauses=[(new[a],rng.choice(forms[1])),(new[b],rng.choice(forms[0]))]
            rng.shuffle(clauses)
            scores=clf.decision_function(vec.transform([s for n,s in clauses]))
            choice=lex[clauses[scores.argmax()][0]]
            counts[0]+=int(choice==a);counts[1]+=int(abs(scores[0]-scores[1])>1e-8);counts[2]+=1
        results[name]=counts
    return dict(support_clauses=len(docs),alias_bindings=len(lex),results=results)
if __name__=='__main__':
    (HERE/'results/grammar-repair.json').write_text(json.dumps({s:run(s) for s in (5101,5107,5113)}))
