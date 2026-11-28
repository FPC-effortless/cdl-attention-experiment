"""Audit exact #119 cognitive task generation; no evaluation on confirmation split."""
import json, numpy as np, collections
from pathlib import Path
from space import BASE
from knowledge import *
from train import SEEDS,STEPS,KB
OUT=Path(__file__).parent

def shortcut(x,default):
    z=x.numpy();event=z[:,-11:-8];suffix=z[:,-8:];q=np.take_along_axis(suffix,(suffix==TARGET).argmax(1)[:,None]+1,axis=1).ravel()+0;ops=suffix[:,-3:-1]-OP
    target=event[:,1]==q;direct=(event[:,0]==WRITE)&target;reset=(event[:,0]==RESET)&target
    pred=np.full(len(z),default,dtype=np.int64);pred[direct]=((event[direct,2]-VAL)+(ops[direct]==1).sum(1)-(ops[direct]==2).sum(1))%4+VAL;pred[reset]=VAL+EMPTY
    return pred,direct,reset,target,event[:,0]

def metrics(x,y,default):
    p,w,r,t,a=shortcut(x,default);y=y.numpy();n=len(y)
    return dict(n=n,write=int((a==WRITE).sum()),hold=int((a==HOLD).sum()),reset=int((a==RESET).sum()),mutation_target=int(t.sum()),answer_directly_available=int((w|r).sum()),direct_accuracy=int(((p==y)&(w|r)).sum()),fallback_accuracy=int(((p==y)&~(w|r)).sum()),shortcut_correct=int((p==y).sum()),answer_hist={str(k):int(v) for k,v in collections.Counter(y.tolist()).items()})

def main():
    rows=[]
    for seed in SEEDS:
        train,dev,_=split(seed,8);g=np.random.default_rng(seed+21203);xs=[];ys=[]
        for _ in range(STEPS):
            x,y=draw8(g,train,KB);xs.append(x);ys.append(y)
        x=torch.cat(xs);y=torch.cat(ys);_,w,r,*_=shortcut(x,VAL);mode=int(collections.Counter(y.numpy()[~(w|r)].tolist()).most_common(1)[0][0]);trainres=metrics(x,y,mode)
        assert trainres['answer_directly_available']==trainres['direct_accuracy']
        rng=np.random.default_rng(seed+8601);n=256;ids=rng.choice(dev,n,replace=False);q=rng.integers(8,size=n);other=(q+rng.integers(1,8,size=n))%8;op=rng.integers(3,size=(n,2));order=np.argsort(rng.random((n,8)),1);syntax=rng.integers(2,4,size=n);new,keep=choose_pair(ids,q,dev,8)
        panels={}
        for panel,horizon,md in [('held_full',0,'full'),('held_oracle',0,'oracle'),('held_blank',0,'blank'),('held_long',64,'full')]:
            hold=np.stack((np.full((n,horizon),HOLD),rng.integers(8,size=(n,horizon))+KEY,np.full((n,horizon),MASK)),-1)
            xx,yy,_=make8(ids,q,op,order,other,syntax,mode=md,hold_events=hold);xx2,yy2,_=make8(new,q,op,order,other,syntax,mode=md,hold_events=hold)
            rr=metrics(xx,yy,mode);p,*_=shortcut(xx,mode);p2,*_=shortcut(xx2,mode);rr.update(pairs=int(keep.sum()),both_correct=int(((p==yy.numpy())&(p2==yy2.numpy())&keep).sum()));panels[panel]=rr
        rows.append(dict(seed=seed,unresolved_training_modal_value_token=mode,training=trainres,development=panels))
    result=dict(status='GENERATOR_AUDIT_NOT_MODEL_SCORE',rows=rows,train_counts_total={k:sum(r['training'][k] for r in rows) for k in ('n','write','hold','reset','mutation_target','answer_directly_available','shortcut_correct')},held_full_total={k:sum(r['development']['held_full'][k] for r in rows) for k in ('n','answer_directly_available','shortcut_correct','pairs','both_correct')})
    (OUT/'objective_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['train_counts_total'],result['held_full_total'])
if __name__=='__main__':main()
