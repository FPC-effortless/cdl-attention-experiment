"""Learn a program under an unknown one-axis measurement change point."""
import random

import numpy as np

from bias import resolve
from epistemic import grammar

SEEDS=(521,547,569)
KINDS=("constant","piecewise","checkerboard")


def conditions(seed):
    rng=random.Random(seed)
    programs,_=grammar()
    matrix=np.array([values for values,_ in programs],dtype=np.int8)
    x,y=np.indices((5,5)).reshape(2,25)
    axis=(x,y)
    conditions=[0*x]+[(coordinate>=threshold).astype(np.int8) for coordinate in axis for threshold in range(1,5)]
    cases=[]
    for kind in KINDS:
        for trial in range(80):
            truth=np.array(rng.choice(programs)[0],dtype=np.int8)
            offset=rng.randrange(5)
            gain=rng.randrange(1,5)
            switch=rng.choice(conditions[1:])
            if kind=="checkerboard":
                switch=(x+y)%2
            if kind=="constant":
                switch=conditions[0]
            measured=(truth+offset+gain*switch)%5
            order=rng.sample(range(25),25)
            cases.append((kind,trial,truth,measured,order[:8],order[8:15],order[15:17],order[17:]))
    return matrix,np.array(conditions),cases


def run(seed,arm):
    matrix,switches,cases=conditions(seed)
    changes=range(5) if arm=="candidate" else (0,)
    model=switches if arm=="candidate" else switches[:1]
    rows=[]
    for kind,trial,truth,measured,train,pool,validation,test in cases:
        remaining=[]
        for switch in model:
            for change in changes:
                # Marginalize an arbitrary constant offset by comparing pairwise measurement residuals.
                residual=(measured[train]-matrix[:,train]-change*switch[train])%5
                remaining.extend(matrix[np.all(residual==residual[:,:1],axis=1)])
        candidates=np.unique(np.asarray(remaining,dtype=np.int8).reshape(-1,25),axis=0)
        for budget in (1,2,4):
            choose=resolve(candidates,pool,truth,truth,[],test,"active",budget)
            survivors=candidates
            for key,label in choose["queries"]:
                survivors=survivors[survivors[:,key]==label]
            # Independent validation is outside the query-selection pool.
            eligible=choose["prediction"] is not None and np.all(survivors[:,validation]==truth[validation])
            rows.append(dict(scenario=kind,trial=trial,budget=budget,validation=bool(eligible),
                             test_truth=truth[test].tolist(),**choose))
    return dict(rows=rows)
