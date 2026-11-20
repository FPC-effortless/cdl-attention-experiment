"""Learn latent narration clusters without class labels and test transfer to subsequent actions."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_mutual_info_score
from kitchen import read,h,R

def main():
    original=read('train');test=read('validation')
    tr=[r for r in original if h(r['video_id'])%7];cal=[r for r in original if not h(r['video_id'])%7]
    rng=np.random.default_rng(5152);cal_ids=rng.choice(len(tr),18000,replace=False)
    train_docs=[tr[i]['narration'] for i in cal_ids]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,sublinear_tf=True)
    z=vec.fit_transform(train_docs)
    svd=TruncatedSVD(n_components=48,random_state=5152).fit(z)
    code=MiniBatchKMeans(n_clusters=64,n_init=3,batch_size=512,random_state=5152).fit(normalize(svd.transform(z)))
    allrows=tr+cal+test
    compressed=code.predict(normalize(svd.transform(vec.transform([r['narration'] for r in allrows]))))
    first=[r['narration'].strip().lower().split()[0] if r['narration'].strip() else '' for r in allrows]
    unseen={r['participant_id'] for r in test}-{r['participant_id'] for r in tr}
    def transitions(rows):
        videos=defaultdict(list)
        for i,r in enumerate(rows):videos[r['video_id']].append((int(r['start_frame']),i))
        return [(a[1],b[1]) for v in videos.values() for a,b in zip(sorted(v),sorted(v)[1:])]
    results={}
    for name,symbols in [('latent',compressed.tolist()),('first_word',first),('constant',['*']*len(first))]:
        cal_rep=symbols[len(tr):len(tr)+len(cal)]
        test_rep=symbols[-len(test):]
        current=defaultdict(Counter);nxt=defaultdict(Counter);prior=Counter()
        for i,r in enumerate(cal):
            y=r['verb_class'];current[cal_rep[i]][y]+=1;prior[y]+=1
        for i,j in transitions(cal):nxt[cal_rep[i]][cal[j]['verb_class']]+=1
        def get(table,rep):return (table.get(rep) or prior).most_common(1)[0][0]
        preds=[get(current,k) for k in test_rep]
        pairs=transitions(test)
        results[name]=dict(current=sum(p==r['verb_class'] for p,r in zip(preds,test)),n=len(test),
            unseen_correct=sum(p==r['verb_class'] for p,r in zip(preds,test) if r['participant_id'] in unseen),
            unseen_n=sum(r['participant_id'] in unseen for r in test),
            next=sum(get(nxt,test_rep[i])==test[j]['verb_class'] for i,j in pairs),n_next=len(pairs),
            ami_verb=adjusted_mutual_info_score([r['verb_class'] for r in test],test_rep),
            ami_noun=adjusted_mutual_info_score([r['noun_class'] for r in test],test_rep))
    (R/'latent-kitchen.json').write_text(json.dumps(results))
if __name__=='__main__':main()
