"""Fixed deterministic multinomial probe with TRAIN-only feature scaling."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

ALPHA=.01; MAX_ITER=300; TOL=1e-9
def req(x,s):
    if not x:raise ValueError(s)

def fit(x, y, evaluate, classes, *, prior=False):
    x=np.asarray(x,dtype=np.float64);z=np.asarray(evaluate,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64)
    req(x.ndim==z.ndim==2 and x.shape[1]==z.shape[1] and len(x)==len(y)>0,'geometry')
    req(set(y)==set(range(classes)) and np.isfinite(x).all() and np.isfinite(z).all(),'unexpected support')
    if prior:
        logp=np.log(np.bincount(y,minlength=classes)/len(y))
        lp=np.broadcast_to(logp,(len(z),classes)).copy()
        return lp,dict(steps=0,train_loss=-float(np.mean(logp[y])),dimension=0,rank=0)
    m=x.mean(0); sd=x.std(0);sd[sd<=1e-8]=1.0
    x=(x-m)/sd;z=(z-m)/sd
    sv=np.linalg.svd(x,compute_uv=False)
    rank=int(sum(sv>sv[0]*1e-6)) if len(sv) else 0
    n,d=x.shape
    t=np.eye(classes,dtype=np.float64)[y]
    def loss(v):
        w=v[:d*classes].reshape(d,classes);b=v[d*classes:]
        logits=x@w+b; lp=logits-logsumexp(logits,axis=1,keepdims=True)
        j=-lp[np.arange(n),y].mean()+.5*ALPHA*float(np.sum(w*w))
        err=(np.exp(lp)-t)/n
        g=np.r_[(x.T@err+ALPHA*w).ravel(),np.sum(err,axis=0)]
        return j,g
    q=minimize(loss,np.zeros((d+1)*classes),jac=True,method='L-BFGS-B',tol=TOL,
               options={'maxiter':MAX_ITER,'ftol':TOL,'gtol':TOL})
    req(bool(q.success) and np.isfinite(q.fun) and np.isfinite(q.x).all(),'fixed probe optimization failed')
    w=q.x[:d*classes].reshape(d,classes);b=q.x[d*classes:]
    logits=z@w+b
    logp=logits-logsumexp(logits,axis=1,keepdims=True)
    req(np.isfinite(logp).all(),'nonfinite probe prediction')
    return logp,dict(steps=int(q.nit),train_loss=float(q.fun),rank=rank,dimension=d)

def metrics(logp,labels,candidates=None):
    y=np.asarray(labels,dtype=np.int64)
    req(logp.ndim==2 and len(logp)==len(y)>0,'score shape')
    p=logp[np.arange(len(y)),y]
    other=logp.copy();other[np.arange(len(y)),y]=-np.inf
    correct=p>other.max(axis=1)
    out=dict(rows=len(y),correct=int(correct.sum()),ce_nats=float(-p.mean()),
             target_top_ties=int((p==other.max(axis=1)).sum()))
    if candidates is not None:
        q=np.asarray(candidates,dtype=np.int64)
        req(q.shape==y.shape and np.all(q!=y),'invalid foil')
        other_p=logp[np.arange(len(y)),q]
        out.update(relative_correct=int((p>other_p).sum()),relative_ce_nats=float(np.logaddexp(0,other_p-p).mean()))
    return out, correct

if __name__=='__main__':
    rng=np.random.default_rng(4)
    y=np.array([0,1]*64); x=rng.normal(0,.1,(128,5));x[:,0]+=2*y-1
    p,r=fit(x,y,x,2)
    m,_=metrics(p,y,1-y)
    assert m['correct']==m['relative_correct']==128 and m['ce_nats']<.1 and r['rank']==5
    p,_=fit(x,y,x,2,prior=True);m,_=metrics(p,y)
    assert m['correct']==0 and abs(m['ce_nats']-np.log(2))<1e-12
    print('inert balanced prior tie and separable probe checks passed')
