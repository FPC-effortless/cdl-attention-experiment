"""New held-world tasks. Four-key input preserves COG-COGNITION-001 grammar
up to an injective symbol remapping. Eight-key events add explicit mutations,
distractor queries, two operations and streamed revision. No key/operation labels
are supplied separately to any model; gold is used only for loss/evaluation.
"""
import itertools,hashlib,math
import numpy as np
import torch
START,MASK,QUERY=256,257,258
KEY,VAL,OP=259,267,272
WRITE,HOLD,RESET,TARGET,DIST,END=275,276,277,278,279,280
EMPTY=4

def split(seed,k):
    rng=np.random.default_rng(seed+10000+997*(k-4));a=rng.permutation(4**k);n=len(a)
    return a[:3*n//4],a[3*n//4:7*n//8],a[7*n//8:]
def maps(ids,k):
    ids=np.asarray(ids,dtype=np.int64);return (ids[:,None]//(4**np.arange(k-1,-1,-1,dtype=np.int64)))%4

def choose_pair(ids,q,allowed,k):
    world=maps(ids,k);old=world[np.arange(len(ids)),q];place=4**(k-1-q)
    options=ids[:,None]+(((old[:,None]+np.arange(1,4))%4)-old[:,None])*place[:,None]
    mask=np.zeros(4**k,dtype=bool);mask[allowed]=True;ok=mask[options];keep=ok.any(1);new=np.where(keep,options[np.arange(len(ids)),ok.argmax(1)],ids)
    assert np.all(mask[new]);return new,keep

def make4(ids,q,op,order,mode='full'):
    w=maps(ids,4);n=len(ids);x=np.full((n,13),MASK,dtype=np.int64);x[:,0]=START;x[:,1:9:2]=order+KEY
    if mode=='full':x[:,2:9:2]=np.take_along_axis(w,order,1)+VAL
    x[:,9]=QUERY;x[:,10]=q+KEY
    if mode=='oracle':x[:,11]=w[np.arange(n),q]+VAL
    x[:,12]=op+OP;y=(w[np.arange(n),q]+op)%4+VAL
    return torch.from_numpy(x),torch.from_numpy(y)

def draw4(rng,allowed,n,fixed=False,mixed=False):
    ids=rng.choice(allowed,n) if not fixed else np.repeat(allowed[:1],n)
    q=rng.integers(4,size=n);op=rng.integers(2,size=n);order=np.argsort(rng.random((n,4)),1)
    x,y=make4(ids,q,op,order)
    if mixed:
        a,_=make4(ids,q,op,order,'oracle');x[n//2:]=a[n//2:]
    return x,y

def make8(ids,q,op,order,other,syntax,*,mode='full',hold_events=None,change=None):
    # change shape [n,3] [operation,key,value]; final query target may be empty.
    n=len(ids);w=maps(ids,8);records=np.stack((np.full_like(order,WRITE),order+KEY,np.take_along_axis(w,order,1)+VAL),-1)
    if hold_events is None:hold_events=np.zeros((n,0,3),np.int64)
    if change is None:change=np.stack((np.full(n,HOLD),other+KEY,np.full(n,MASK)),-1)
    w2=w.copy();i=np.arange(n);k=change[:,1]-KEY;iswrite=change[:,0]==WRITE;isreset=change[:,0]==RESET
    w2[i[iswrite],k[iswrite]]=change[iswrite,2]-VAL;w2[i[isreset],k[isreset]]=EMPTY
    value=w2[i,q];answer=np.where(value==EMPTY,EMPTY,(value+(op==1).sum(1)-(op==2).sum(1))%4)+VAL
    prefix=np.concatenate((np.full((n,1),START),records.reshape(n,-1),hold_events.reshape(n,-1),change),1)
    if mode in ('oracle','blank'):
        prefix[:,1:]=MASK
        if mode=='oracle':
            prefix[:,1]=np.where(value==EMPTY,RESET,WRITE);prefix[:,2]=q+KEY;prefix[:,3]=np.where(value==EMPTY,MASK,value+VAL)
    a=np.stack((np.full(n,QUERY),np.full(n,TARGET),q+KEY,np.full(n,DIST),other+KEY,op[:,0]+OP,op[:,1]+OP,np.full(n,END)),-1)
    b=np.stack((np.full(n,QUERY),np.full(n,DIST),other+KEY,np.full(n,TARGET),q+KEY,op[:,0]+OP,op[:,1]+OP,np.full(n,END)),-1)
    c=np.stack((np.full(n,DIST),other+KEY,np.full(n,TARGET),q+KEY,np.full(n,QUERY),op[:,0]+OP,op[:,1]+OP,np.full(n,END)),-1)
    d=np.stack((np.full(n,TARGET),q+KEY,np.full(n,DIST),other+KEY,np.full(n,QUERY),op[:,0]+OP,op[:,1]+OP,np.full(n,END)),-1)
    query=np.stack((a,b,c,d),1)[i,syntax]
    return torch.from_numpy(np.concatenate((prefix,query),1)),torch.from_numpy(answer),prefix.shape[1]

def draw8(rng,allowed,n,fixed=False,mixed=False):
    ids=rng.choice(allowed,n) if not fixed else np.repeat(allowed[:1],n)
    q=rng.integers(8,size=n);other=(q+rng.integers(1,8,size=n))%8;op=rng.integers(3,size=(n,2));order=np.argsort(rng.random((n,8)),1);syntax=rng.integers(2,size=n)
    new,keep=choose_pair(ids,q,allowed,8);v=maps(new if not fixed else ids,8)[np.arange(n),q]
    actions=rng.choice([WRITE,HOLD,RESET],n,p=[.4,.4,.2]);actions[(actions==WRITE)&~keep]=HOLD
    ch=np.stack((actions,q+KEY,np.where(actions==WRITE,v+VAL,MASK)),-1)
    x,y,_=make8(ids,q,op,order,other,syntax,change=ch)
    if mixed:
        a,_,_=make8(ids,q,op,order,other,syntax,mode='oracle',change=ch);x[n//2:]=a[n//2:]
    return x,y

def digest_split(seed):
    return {str(k):[hashlib.sha256(x.tobytes()).hexdigest() for x in split(seed,k)] for k in (4,8)}

@torch.no_grad()
def eval4(m,seed):
    _,dev,_=split(seed,4);rng=np.random.default_rng(seed+8501);ids=np.repeat(dev,8);q=np.tile(np.repeat(np.arange(4),2),len(dev));op=np.tile([0,1],len(ids)//2);order=np.argsort(rng.random((len(ids),4)),1);new,keep=choose_pair(ids,q,dev,4);r={}
    for mode in ('full','oracle','blank'):
        x,y=make4(ids,q,op,order,mode);xx,yy=make4(new,q,op,order,mode);p=predict(m,x);p2=predict(m,xx);ok=torch.from_numpy(keep);b=(p==y)&(p2==yy)&ok
        r[mode]=dict(n=len(x),correct=int((p==y).sum()),pairs=int(ok.sum()),both_correct=int(b.sum()),valid=int(((p>=VAL)&(p<VAL+4)).sum()))
    return r

def predict(m,x):return torch.cat([m(z)[:,-1].argmax(-1) for z in x.split(16)])

@torch.no_grad()
def eval8(m,seed,n=256):
    _,dev,_=split(seed,8);rng=np.random.default_rng(seed+8601);ids=rng.choice(dev,n,replace=False);q=rng.integers(8,size=n);other=(q+rng.integers(1,8,size=n))%8;op=rng.integers(3,size=(n,2));order=np.argsort(rng.random((n,8)),1);syntax=rng.integers(2,4,size=n);new,keep=choose_pair(ids,q,dev,8);out={}
    for panel,horizon,mode in [('held_full',0,'full'),('held_oracle',0,'oracle'),('held_blank',0,'blank'),('held_long',64,'full')]:
        hold=np.stack((np.full((n,horizon),HOLD),rng.integers(8,size=(n,horizon))+KEY,np.full((n,horizon),MASK)),-1)
        x,y,cut=make8(ids,q,op,order,other,syntax,mode=mode,hold_events=hold);xx,yy,_=make8(new,q,op,order,other,syntax,mode=mode,hold_events=hold);p=predict(m,x);p2=predict(m,xx);ok=torch.from_numpy(keep)
        out[panel]=dict(n=n,correct=int((p==y).sum()),pairs=int(ok.sum()),both_correct=int(((p==y)&(p2==yy)&ok).sum()))
    # Actual stream revision: process one knowledge prefix, append a changed fact,
    # query identical suffix; compare with a fresh replacement world and preserve model weights.
    old,y,cut=make8(ids,q,op,order,other,syntax);updated,yy,_=make8(new,q,op,order,other,syntax)
    vv=maps(new,8)[np.arange(n),q];event=torch.from_numpy(np.stack((np.full(n,WRITE),q+KEY,vv+VAL),-1));reset=torch.from_numpy(np.stack((np.full(n,RESET),q+KEY,np.full(n,MASK)),-1))
    p1=[];p2=[];pr=[];rs=[];parity=0;non_target=0;non_target_n=0
    # Semantic target is not an internal slot index. No unsupported semantic-slot claim.
    for i in range(0,n,16):
        x=old[i:i+16];suff=x[:,cut:];_,s=m(x[:,:cut],carry=True);z=m(suff,s)[:,-1];p1.append(z.argmax(-1));parity+=int((z.argmax(-1)==m(x)[:,-1].argmax(-1)).sum());_,s2=m(event[i:i+16],s,True);pr.append(m(suff,s2)[:,-1].argmax(-1));_,sr=m(reset[i:i+16],s,True);rs.append(m(suff,sr)[:,-1].argmax(-1));p2.append(m(updated[i:i+16])[:,-1].argmax(-1))
    a,b,c,d=map(torch.cat,(p1,p2,pr,rs));ok=torch.from_numpy(keep)
    out['revision']=dict(n=n,pairs=int(ok.sum()),both_correct=int(((a==y)&(c==yy)&ok).sum()),updated_correct=int(((c==yy)&ok).sum()),fresh_replacement_agreement=int(((c==b)&ok).sum()),cached_full_agreement=parity,reset_correct=int((d==VAL+EMPTY).sum()))
    return out
