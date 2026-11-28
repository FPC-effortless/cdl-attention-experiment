import collections,json,hashlib
from pathlib import Path
import numpy as np,torch
from space import BASE
from train import SEEDS,STEPS,KB
from knowledge import split,VAL
from balanced import generate,independent_answer,counterfactuals
from audit_objective import shortcut,metrics
OUT=Path(__file__).parent

def main():
    rows=[]
    for s in SEEDS:
        train,dev,reserved=split(s,8);assert not set(train)&set(dev) and not set(train)&set(reserved) and not set(dev)&set(reserved)
        def stream():
            g=np.random.default_rng(s+21203);xs=[];ys=[]
            for _ in range(STEPS):
                x,y,m=generate(g,train,KB);assert torch.equal(independent_answer(x),y);assert set(m['ids'])<=trainset and set(m['post'])<=trainset
                xs.append(x);ys.append(y)
            return torch.cat(xs),torch.cat(ys)
        trainset=set(train.tolist());x,y=stream();xx,yy=stream();assert torch.equal(x,xx) and torch.equal(y,yy)
        _,w,r,*_=shortcut(x,VAL);default=int(collections.Counter(y.numpy()[~(w|r)].tolist()).most_common(1)[0][0]);tr=metrics(x,y,default)
        x,y,m=generate(np.random.default_rng(s+8601),dev,256,True);assert torch.equal(independent_answer(x),y);assert set(m['ids'])<=set(dev) and set(m['post'])<=set(dev)
        ix,x2,y2,eligible=counterfactuals(x,y,m,dev);p,*_=shortcut(x,default);p2,*_=shortcut(x2,default);held=metrics(x,y,default);held.update(prefix_dependent_eligible=eligible,pairs=len(ix),both_correct=int(((p[ix]==y[ix].numpy())&(p2==y2.numpy())).sum()))
        assert tr['answer_directly_available']==tr['n']//4 and held['answer_directly_available']==held['n']//4 and held['pairs']>0 and held['both_correct']==0
        rows.append(dict(seed=s,training=tr,held=held,deterministic_replay=True,all_train_and_held_answers_match_independent_interpreter=True))
        print(s,tr['shortcut_correct'],held['shortcut_correct'],held['pairs'],flush=True)
    d=dict(status='GENERATOR_AND_SYMBOLIC_ORACLE_ONLY',rows=rows,train_total={k:sum(r['training'][k] for r in rows) for k in ('n','mutation_target','answer_directly_available','shortcut_correct')},held_total={k:sum(r['held'][k] for r in rows) for k in ('n','answer_directly_available','shortcut_correct','prefix_dependent_eligible','pairs','both_correct')},reserved_world_examples_evaluated=0)
    (OUT/'balanced_verification.json').write_text(json.dumps(d,indent=2)+'\n');print(d['train_total'],d['held_total'])
if __name__=='__main__':main()
