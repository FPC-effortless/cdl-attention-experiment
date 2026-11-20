"""Authentic language and state-transfer diagnostics, with upstream bytes pinned."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.request import urlopen

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'cog-attn-handoff/continuation/cog-051-060'
sys.path[:0]=[str(ROOT/'vendor'),str(BASE)]
from expiry import Runtime
import dialogue_transfer as original

def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def vectors(training,test,expanded):
    vec=[TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1)]
    if expanded:vec.append(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True))
    for v in vec:v.fit(training)
    return vec, [hstack([v.transform(texts) for v in vec],format='csr') for texts in (training,test)]

def preserve(dataset,arm,vec,model):
    if os.environ.get('COG_CHECKPOINT_DIR'):
        import joblib
        base=Path(os.environ['COG_CHECKPOINT_DIR'])/dataset
        base.mkdir(parents=True,exist_ok=True)
        joblib.dump((vec,model),base/(arm+'.joblib'),compress=3)

def banking():
    pin='57ec275d8078af65b7731c2a98be812d844a6d6b'
    source={};splits={}
    for split in ('train','test'):
        data=urlopen(f'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{pin}/banking_data/{split}.csv',timeout=60).read()
        source[split]=digest(data);splits[split]=list(csv.DictReader(io.StringIO(data.decode())))
    def fold(row):return int(digest(row['text'].strip().lower().encode()),16)%5
    train=[r for r in splits['train'] if fold(r)!=0];cal=[r for r in splits['train'] if fold(r)==0]
    held=splits['test'];results={}
    overlap={r['text'].strip().lower() for r in splits['train']}
    for arm in ('word','expanded'):
        start=perf_counter()
        vec,(x,z)=vectors([r['text'] for r in train],[r['text'] for r in cal+held],arm=='expanded')
        svc=LinearSVC(C=2,random_state=71).fit(x,[r['category'] for r in train])
        preserve('banking',arm,vec,svc)
        scores=svc.decision_function(z);ix=np.argsort(-scores,axis=1,kind='stable')[:,:5]
        order=svc.classes_[ix];margin=np.take_along_axis(scores,ix[:,:2],axis=1);margin=margin[:,0]-margin[:,1]
        gold=np.array([r['category'] for r in cal+held]);n=len(cal)
        # Freeze a grid-calibrated selective threshold at empirical .99.
        cuts=[t for t in np.arange(0,3,.05) if (margin[:n]>=t).any() and (order[:n,0]==gold[:n])[margin[:n]>=t].mean()>=.99]
        cut=min(cuts,default=1e6);rows=[]
        for i,r in enumerate(held,n):
            rows.append([digest(r['text'].encode()),str(gold[i]),order[i].tolist(),bool(margin[i]>=cut),r['text'].strip().lower() in overlap])
        results[arm]=dict(seconds=perf_counter()-start,cut=cut,rows=rows)
    return dict(source=source,ntrain=len(train),ncal=len(cal),arms=results)

def issues():
    a=Path('/workspace/issue-history-study/recipe/dataset.json');b=Path('/workspace/issue-history-followup/recipe/dataset.json')
    old=read(a);fresh=[r for r in read(b) if r['strict']]
    train=[r for r in old if r['split']=='train'];cal=[r for r in old if r['split']=='validation' and r['strict']]
    assert {r['issue'] for r in train+cal}.isdisjoint({r['issue'] for r in fresh})
    outcomes={}
    for arm in ('word','expanded'):
        vec,(x,z)=vectors([r['text'] for r in train],[r['text'] for r in cal+fresh],arm=='expanded')
        model=LogisticRegression(C=8,max_iter=350).fit(x,[r['target'] for r in train])
        preserve('issues',arm,vec,model)
        p=model.predict_proba(z)[:,1];truth=np.array([r['target'] for r in cal])
        ts=[t for t in np.arange(.01,1,.01) if (p[:len(cal)]>=t).any() and truth[p[:len(cal)]>=t].mean()>=.99]
        t=min(ts,default=2.)
        rows=[[r['issue'],r['comment'],r['split'],r['target'],float(s),bool(s>=t)] for r,s in zip(fresh,p[len(cal):])]
        outcomes[arm]=dict(threshold=t,rows=rows)
    return dict(source=[digest(p.read_bytes()) for p in (a,b)],arms=outcomes)

def dialogue():
    verify=read(BASE/'results/dialogue-transfer.json')
    hashes={};rows=original.corpus('test',hashes)
    assert hashes=={k:v for k,v in verify['source'].items() if k.startswith('test/')}
    vec,models=joblib.load(BASE/'checkpoints/dialogue/models.joblib')
    x=hstack([v.transform([r[2][field] for r in rows]) for field,v in vec],format='csr')
    probs=[models[k][0].predict_proba(x) for k in original.SLOTS]
    labels={k:models[k][0].classes_.tolist() for k in original.SLOTS}
    gates={k:verify['slots'][k+'/direct']['gated']['threshold'] for k in original.SLOTS}
    output={}
    for arm in ('verified','soft','expiry2','expiry4','repair'):
        last=None;tr=[];state=None
        for i,(d,t,text,delta) in enumerate(rows):
            if last!=d:
                if state:state.close()
                state=Runtime(labels,gates,'soft' if arm=='repair' else arm,'locked')
                truth=['<empty>']*2;last=d;previous=None
            ps=[p[i] for p in probs]
            old=[m.copy() for m in state.r.mass]
            age=state.observe(t,ps)
            if arm=='repair':
                repeated=text[0]==previous
                correction=bool(re.search(r'\b(actually|instead|rather|sorry|correction)\b',text[0]))
                for j,k in enumerate(labels):
                    if k in age['patch']:continue
                    if repeated:state.r.mass[j]=old[j]
                    elif correction and labels[k][ps[j].argmax()]!='<hold>':
                        new=ps[j].copy();new[labels[k].index('<hold>')]=0;state.r.mass[j]=new/new.sum()
                age['read']=[v[0] for v in state.r.estimates().values()]
            truth=[v if v!='<hold>' else o for o,v in zip(truth,delta)]
            tr.append([d,t,age['read']==truth,all(v=='<hold>' for v in delta),age['patch'],age['read']])
            previous=text[0]
        state.close();output[arm]=tr
    assert all([r[4] for r in tr]==[r[4] for r in output['soft']] for tr in output.values())
    assert sum(not r[2] and r[3] for r in output['soft'])==665
    return output
if __name__=='__main__':
    for name in sys.argv[1:] or ('banking','issues','dialogue'):
        (ROOT/'results'/('authentic-'+name+'.json')).write_text(json.dumps(globals()[name](),separators=(',',':')))
