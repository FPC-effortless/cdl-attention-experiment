"""Two same-budget re-rankers with identical text, different history order."""
import warnings
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

WIDTH=3*8192+2+2*8

def ordered_text(row, mode):
    current,*history=row["text"]
    if mode=="canonical":
        history.sort()
    elif mode!="ordered":
        raise ValueError("unknown history mode")
    return [current,*history]

def base_features(rows, existing=None):
    vectors=[] if existing is None else existing
    blocks=[]
    n=0
    for field in (0,1):
        docs=[r["text"][0] if field==0 else " ".join(r["text"][1:]).strip() for r in rows]
        for analyzer,grams in (("word",(1,2)),("char",(3,5))):
            if existing is None:
                v=TfidfVectorizer(analyzer=analyzer,ngram_range=grams,min_df=5,max_features=32000,sublinear_tf=True)
                v.fit(docs);vectors.append(v)
            blocks.append(vectors[n].transform(docs));n+=1
    return hstack(blocks,format="csr"),vectors

def shortlist(p):
    if p.ndim!=2 or p.shape[1]<2 or p.shape[1]>8 or not np.isfinite(p).all():
        raise ValueError("invalid proposal probabilities")
    ids=np.argsort(-p,axis=1,kind="stable")[:,:2]
    return ids,np.take_along_axis(p,ids,axis=1)

def meta_text(rows,mode):
    docs=[ordered_text(r,mode) for r in rows]
    blocks=[HashingVectorizer(n_features=8192,ngram_range=(1,2),alternate_sign=True,dtype=np.float64).transform([d[i] for d in docs]) for i in range(3)]
    return blocks

def meta_features(blocks,ids,prob):
    n,k=ids.shape
    if k!=2 or prob.shape!=(n,2): raise ValueError("bad shortlist")
    pos=np.zeros((n,16))
    pos[np.arange(n),ids[:,0]]=1.
    pos[np.arange(n),8+ids[:,1]]=1.
    x=hstack([*blocks,csr_matrix(np.column_stack((prob,pos)))],format="csr")
    if x.shape!=(n,WIDTH): raise ValueError("model width mismatch")
    return x

def classifier(x,y,max_iter,c):
    m=LogisticRegression(max_iter=max_iter,C=c,solver="lbfgs",random_state=62062)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always",ConvergenceWarning)
        m.fit(x,y)
    if not np.isfinite(m.coef_).all() or not np.isfinite(m.intercept_).all():
        raise ValueError("nonfinite fit")
    return m,not any(issubclass(w.category,ConvergenceWarning) for w in ws)

def rank_targets(labels,choices):
    return np.where(labels==choices[:,0],0,np.where(labels==choices[:,1],1,2))
