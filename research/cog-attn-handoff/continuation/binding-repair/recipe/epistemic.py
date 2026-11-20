"""Silent shifts with controlled observation access and bounded program discovery."""
import itertools
import random
import numpy as np

WIDTH=1024
SCHEDULE=32


def shift_trial(seed,mode,channel):
    rng=random.Random(seed)
    rows=[]
    for world in range(64):
        change=rng.randrange(64,WIDTH-64)
        # Pre-draw noise so all policies receive the same samples at shared audits.
        noise=[[rng.random()<.01 for _ in range(3)] for _ in range(WIDTH//SCHEDULE)]
        observed=False
        wrong=audits=0
        detected=None
        for t in range(WIDTH):
            truth=t>=change
            if mode!="event_only" and t%SCHEDULE==0:
                audits+=3
                compromised=channel=="shared" and mode=="periodic"
                readings=[(truth and not compromised)^error for error in noise[t//SCHEDULE]]
                updated=sum(readings)>=2
                if updated and truth and not observed and detected is None:
                    detected=t-change
                observed=updated
            wrong+=observed!=truth
        rows.append(dict(world=world,change=change,wrong=wrong,anchors=audits,lag=detected))
    return rows


def grammar():
    grid=list(itertools.product(range(5),repeat=2))
    x,y=np.array(grid).T
    leaves={"x":tuple(x),"y":tuple(y),"1":(1,)*25,"2":(2,)*25}
    known={v:k for k,v in leaves.items()}
    counts=[len(known)]
    for depth in range(2):
        last=list(known.items())
        for i,(a,ae) in enumerate(last):
            for b,be in last[i:]:
                for symbol,values in (("+",tuple((u+v)%5 for u,v in zip(a,b))),
                                      ("*",tuple((u*v)%5 for u,v in zip(a,b)))):
                    known.setdefault(values,f"({ae}{symbol}{be})")
        counts.append(len(known))
    return list(known.items()),counts


def discovery(seed):
    rng=random.Random(seed)
    programs,counts=grammar()
    matrix=np.array([k for k,v in programs],dtype=np.int8)
    rows=[]
    for trial in range(120):
        kind=("compositional","random","biased")[trial%3]
        truth=np.array(rng.choice(programs)[0] if kind!="random" else [rng.randrange(5) for _ in range(25)])
        order=rng.sample(range(25),25)
        train,validation,test=order[:10],order[10:16],order[16:]
        observed=(truth+1)%5 if kind=="biased" else truth
        survived=matrix[np.all(matrix[:,train]==observed[train],axis=1)]
        passive=survived[np.all(survived[:,validation]==observed[validation],axis=1)]
        # Intervention labels for the same validation inputs come from an independent channel.
        active=survived[np.all(survived[:,validation]==truth[validation],axis=1)]
        for label,subset in [("shared_validation",passive),("independent_validation",active)]:
            predicted=subset[0] if len(subset) else None
            # Promote only when the bounded grammar uniquely specifies unseen predictions.
            admit=bool(len(subset) and np.all(subset[:,test]==predicted[test]))
            rows.append(dict(trial=trial,kind=kind,mode=label,remaining=len(subset),admit=admit,
                             correct=bool(admit and np.array_equal(predicted[test],truth[test]))))
    return dict(grammar_size=counts,rows=rows)


def run(seed,arm):
    modes=("intervention","periodic") if arm=="candidate" else ("event_only",)
    return dict(shifts=[dict(mode=m,channel=c,worlds=shift_trial(seed,m,c))
                        for m in modes for c in ("independent","shared")],
                discovery=discovery(seed) if arm=="candidate" else None)
