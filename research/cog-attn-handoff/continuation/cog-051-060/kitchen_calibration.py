"""Additional person-disjoint repair test on the official training source."""
from collections import Counter
import json
import numpy as np
from scipy.stats import beta
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from kitchen import read,h,fit,predict,R
def main():
    rows=read('train')
    # Precommit to a hashed subject assignment. No subject crosses partitions within this fit.
    people=sorted({r['participant_id'] for r in rows},key=lambda s:h('new-person-audit-052/'+s))
    held=set(people[:8]);validation=set(people[8:16])
    train=[r for r in rows if r['participant_id'] not in held|validation]
    cal=[r for r in rows if r['participant_id'] in validation]
    test=[r for r in rows if r['participant_id'] in held]
    rng=np.random.default_rng(52733)
    train=[train[i] for i in rng.choice(len(train),18000,replace=False)]
    text=[r['narration'] for r in train+cal+test]
    vec=[TfidfVectorizer(ngram_range=(1,2),max_features=9000,min_df=2),TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=13000,min_df=3)]
    n=len(train);z=len(cal)
    x=hstack([v.fit(text[:n]).transform(text) for v in vec],format='csr')
    truth=np.array([[int(r[k]) for k in ('verb_class','noun_class')] for r in train+cal+test])
    results=[predict(fit(x[:n],truth[:n,i]),x[n:]) for i in range(2)]
    pred=np.stack([r[0] for r in results],axis=1)
    p=results[0][1]*results[1][1]
    ok=(pred==truth[n:]).all(1)
    gates={}
    for arm in ('empirical','finite_sample'):
        choices=[]
        for t in np.arange(.01,1,.01):
            sel=p[:z]>=t
            tp=int(ok[:z][sel].sum());admitted=int(sel.sum())
            bound=tp/admitted if admitted else 0
            if arm=='finite_sample':bound=float(beta.ppf(.05,tp,admitted-tp+1)) if tp else 0
            if bound>=.99:choices.append(t)
        t=min(choices,default=2);take=p[z:]>=t
        groups={}
        for r,yes,correct in zip(test,take,ok[z:]):
            if yes:groups[r['video_id']]=groups.get(r['video_id'],True) and correct
        good=sum(groups.values());tp=int(ok[z:][take].sum())
        gates[arm]=dict(threshold=float(t),held=len(test),admitted=int(take.sum()),tp=tp,
                        recall=tp/len(test),precision=tp/int(take.sum()) if take.any() else None,
                        video_clean_lower=float(beta.ppf(.05,good,len(groups)-good+1)) if good else 0)
    (R/'kitchen-calibration.json').write_text(json.dumps(dict(training=n,cal=z,held_people=sorted(held),cal_people=sorted(validation),gates=gates)))
if __name__=='__main__':main()
