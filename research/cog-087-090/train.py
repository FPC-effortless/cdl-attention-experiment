"""Learn skill preconditions, effects and a lightweight remaining-cost predictor."""
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tasks import tasks,encoding,acquire,scramble_options,DOMAINS
from skills import suffix,features
from search import solve

ROOT=Path(__file__).resolve().parent
def states(t,actions):
    s=t.start;yield s
    for a in actions:
        s=dict(t.neighbors(s))[a];yield s

def fit():
    started=perf_counter_ns();train=tasks('train');dev=tasks('dev')
    refs={t.id:solve(t) for t in train};book={k:{} for k in DOMAINS};traces=defaultdict(list)
    evidence=[];corridors=0
    for t in train:
        path=refs[t.id]['actions'];ss=list(states(t,path))
        traces[t.kind].append(dict(verified=True,path=path,source_family=t.digest()))
        for j,s in enumerate(ss[:-1]):
            tail=path[j:]
            if t.kind=='maze':
                if len(t.neighbors(s))!=2 and j+2<len(ss) and len(t.neighbors(ss[j+1]))==2:
                    end=j+1
                    while end<len(path) and len(t.neighbors(ss[end]))==2:end+=1
                    evidence.append([t.id,j,'corridor',list(s),list(ss[end]),end-j]);corridors+=1
                continue
            if t.kind=='slide':
                settled=any(v and v==g and v!=p for v,g,p in zip(s,t.target,ss[j-1] if j else s))
            else:
                settled=len(s)>1 and s[0]==s[1] and (not j or ss[j-1][0]!=ss[j-1][1])
            if not settled:continue
            k,ids=encoding(t,s)
            seq=tail if t.kind=='slide' else tuple((ids[a],ids[b]) for a,b in tail)
            prev=book[t.kind].get(k)
            if prev is None or len(tail)<len(prev):book[t.kind][k]=seq
            evidence.append([t.id,j,'settled_suffix',list(s),list(t.target),len(seq)])
    # Disjoint world keys for target estimation: leave each seed's demonstrations out.
    def other_book(t):
        b={k:{} for k in DOMAINS}
        for ident,j,typ,_,_,_ in evidence:
            if typ!='settled_suffix' or ident.split('/')[1]==t.id.split('/')[1]:continue
            u=lookup[ident];path=refs[ident]['actions'];s2=state_paths[ident][j]
            key,ids=encoding(u,s2);tail=path[j:]
            val=tail if u.kind=='slide' else tuple((ids[a],ids[z]) for a,z in tail)
            if key not in b[u.kind] or len(val)<len(b[u.kind][key]):b[u.kind][key]=val
        return b
    lookup={t.id:t for t in train};state_paths={t.id:list(states(t,refs[t.id]['actions'])) for t in train}
    books={seed:other_book(next(t for t in train if t.id.split('/')[1]==seed)) for seed in {t.id.split('/')[1] for t in train}}
    rows=[]
    for t in train:
        m=dict(book=books[t.id.split('/')[1]],corridor_support=corridors)
        samples=state_paths[t.id][:-1]
        indices=sorted(set(np.linspace(0,len(samples)-1,min(8,len(samples)),dtype=int)))
        for j in indices:
            s=samples[j];u=replace(t,start=s)
            seq=() if t.kind=='maze' else suffix(t,s,m['book'])
            a=solve(u);b=solve(u,'structural',model=m,first=True)
            assert len(a['actions'])==len(b['actions'])
            rows.append(dict(source=t.id,state=list(s),features=features(t,s,t.neighbors(s),seq),
                             baseline=a['attempts'],invoked=b['attempts'],applicable=bool(b['certified']),
                             delta=a['attempts']-b['attempts']))
    reg=DecisionTreeRegressor(max_depth=4,min_samples_leaf=5,random_state=87)
    reg.fit([r['features'] for r in rows],[r['delta'] for r in rows])
    app=DecisionTreeRegressor(max_depth=4,min_samples_leaf=5,random_state=87).fit(
        [r['features'] for r in rows],[float(r['applicable']) for r in rows])
    def node(tr,i):
        if tr.children_left[i]==-1:return dict(leaf=float(tr.value[i].item()))
        return dict(feature=int(tr.feature[i]),threshold=float(tr.threshold[i]),left=node(tr,tr.children_left[i]),right=node(tr,tr.children_right[i]))
    opt={d:acquire(traces[d]) for d in DOMAINS}
    model=dict(book=book,tree=node(reg.tree_,0),app_tree=node(app.tree_,0),corridor_support=corridors,options=opt,
               shuffled=scramble_options(opt),train_digests=[t.digest() for t in train],
               dev_digests=[t.digest() for t in dev])
    assert len(set(model['train_digests']+model['dev_digests']))==len(train)+len(dev)
    profile={};gates={}
    for d in DOMAINS:
        subset=[t for t in dev if t.kind==d]
        a=[solve(t) for t in subset];b=[solve(t,opts=opt[d]) for t in subset]
        assert [x['steps'] for x in a]==[x['steps'] for x in b]
        gates[d]=bool(opt[d]) and all(sum(r[k] for r in b)<sum(r[k] for r in a) for k in ('attempts','ns'))
        profile[d]=[]
        for tau in (0,.5,2,5,10,10**9):
            v=[solve(t,'E',model=model,tau=tau) for t in subset]
            assert [x['steps'] for x in v]==[x['steps'] for x in a]
            profile[d].append(dict(tau=tau,attempts=sum(r['attempts'] for r in v),ns=sum(r['ns'] for r in v),charged=sum(r['charged'] for r in v)))
    # Optimize the specified primitive-evaluation proxy. Charge machinery separately on test.
    model.update(gates=gates,tau={d:min(v,key=lambda r:(r['attempts'],-r['tau']))['tau'] for d,v in profile.items()})
    out=ROOT/'results'
    for name,obj in [('model',model),('training',dict(samples=rows,tuples=evidence,development=profile,setup_ns=perf_counter_ns()-started))]:
        (out/(name+'.json')).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
if __name__=='__main__':fit()
