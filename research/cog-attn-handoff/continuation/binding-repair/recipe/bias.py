"""Recover a finite program and a measurement offset using separate training anchors."""
import random

import numpy as np

from epistemic import grammar

BUDGETS = (0,1,2,4,8)
KINDS = ("constant","input_dependent","random")
CHANNELS = ("independent","compromised")


def resolve(candidates, pool, observed, truth, queries, test, policy, budget):
    survivors = candidates
    evidence=[]
    pool = list(pool)
    for _ in range(budget):
        if not len(survivors):
            break
        if policy=="active":
            # Prefer the split that minimizes the expected remaining count.
            counts=np.stack([(survivors[:,pool]==label).sum(0) for label in range(5)])
            key=pool[int((counts*counts).sum(0).argmin())]
        else:
            key=next(point for point in queries if point in pool)
        pool.remove(key)
        label=int(observed[key])
        evidence.append((key,label))
        survivors=survivors[survivors[:,key]==label]

    check=pool+list(test)
    unique=bool(len(survivors) and np.all(survivors[:,check]==survivors[0,check]))
    prediction=survivors[0,test].tolist() if unique else None
    return dict(queries=evidence,count=len(survivors),prediction=prediction,
                correct=bool(unique and np.array_equal(prediction,truth[test])))


def run(seed,arm):
    rng=random.Random(seed)
    programs,level_counts=grammar()
    matrix=np.array([values for values,_ in programs],dtype=np.int8)
    rows=[]
    for scenario in KINDS:
        for trial in range(80):
            truth=np.array(rng.choice(programs)[0] if scenario!="random" else [rng.randrange(5) for _ in range(25)],dtype=np.int8)
            offset=rng.randrange(5)
            order=rng.sample(range(25),25)
            train,pool,test=order[:8],order[8:17],order[17:]
            queries=rng.sample(pool,len(pool))
            bias=offset+(np.arange(25)%5>=3) if scenario=="input_dependent" else offset
            dev=(truth+bias)%5
            offsets=range(5) if arm=="candidate" else (0,)
            consistent=[matrix[np.all((matrix[:,train]+change)%5==dev[train],axis=1)] for change in offsets]
            survivors=np.unique(np.concatenate(consistent),axis=0)
            for channel in CHANNELS:
                feedback=truth if channel=="independent" else dev
                for policy in (("active","random") if arm=="candidate" else ("uncorrected",)):
                    for budget in BUDGETS:
                        rows.append(dict(scenario=scenario,trial=trial,channel=channel,policy=policy,budget=budget,
                            initial=len(survivors),**resolve(survivors,pool,feedback,truth,queries,test,policy,budget)))
    return dict(level_counts=level_counts,rows=rows)
