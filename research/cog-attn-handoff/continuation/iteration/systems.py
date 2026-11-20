from array import array
from bisect import bisect_right, insort
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from temporal_base import History, one_dimension, scan


class BucketIndex:
    width = 256

    def __init__(self):
        self.buckets = {}
        self.prefix = {}

    def insert(self, key, valid, tx, value):
        label = key, valid // self.width
        if label not in self.buckets:
            self.buckets[label] = tuple(array('I') for _ in range(3))
        times, dates, values = self.buckets[label]
        times.append(tx)
        dates.append(valid)
        values.append(value)
        if label in self.prefix:
            prefix = self.prefix[label]
            prefix.append(len(dates)-1 if valid >= dates[prefix[-1]] else prefix[-1])

    def lookup(self, key, valid, cutoff):
        for bucket in range(valid // self.width, -1, -1):
            label = key, bucket
            if label not in self.buckets:
                continue
            times, dates, values = self.buckets[label]
            end = bisect_right(times, cutoff)
            if not end:
                continue
            if len(times) >= 64 and valid >= (bucket+1)*self.width-1:
                if label not in self.prefix:
                    prefix = array('I')
                    for i, date in enumerate(dates):
                        prefix.append(i if not prefix or date >= dates[prefix[-1]] else prefix[-1])
                    self.prefix[label] = prefix
                return values[self.prefix[label][end-1]]
            best = -1
            for i in range(end):
                if dates[i] <= valid and (best < 0 or dates[i] >= dates[best]):
                    best = i
            if best >= 0:
                return values[best]
        return -1

    def payload(self):
        arrays = [a for group in self.buckets.values() for a in group] + list(self.prefix.values())
        return sum(len(a)*a.itemsize for a in arrays)


def memory(seed):
    rng = np.random.default_rng(seed)
    size = 32768
    events = []
    for tx in range(size):
        key = 0 if rng.random() < .75 else int(rng.integers(1,32))
        valid = max(0,tx-int(rng.integers(1,size//2))) if rng.random()<.30 else tx
        events.append((key,valid,tx,tx))
    queries = [(0 if rng.random()<.75 else int(rng.integers(1,32)),
                int(rng.integers(size)),int(rng.integers(size))) for _ in range(4096)]
    rows = []
    for mode in ('list','tree','bucket'):
        ordered = [[] for _ in range(32)]
        tree, bucket = History(size), BucketIndex()
        start = perf_counter()
        for k,v,t,y in events:
            if mode=='list':
                insort(ordered[k],(v,t,y))
            elif mode=='tree':
                tree.insert(k,v,t,y)
            else:
                bucket.insert(k,v,t,y)
        build = perf_counter()-start
        result = []
        previous = 0
        elapsed = 0
        for q in [128,256,512,768,1024,1536,2048,4096]:
            start = perf_counter()
            for query in queries[previous:q]:
                if mode=='list':
                    value = one_dimension(ordered,*query)[0]
                elif mode=='tree':
                    value = tree.lookup(*query)[0]
                else:
                    value = bucket.lookup(*query)
                result.append(value)
            elapsed += perf_counter()-start
            previous = q
            rows.append(dict(seed=seed,mode=mode,q=q,build=build,total=build+elapsed,
                             bytes={'bucket':bucket.payload(),'tree':tree.payload(),'list':12*size}[mode]))
        if mode=='list':
            reference = result
            assert all(reference[i]==scan(events,*queries[i]) for i in range(0,len(queries),32))
        else:
            assert result==reference
    return rows


def verification(seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(2,size=8192)
    rows = []
    # Each provider family is read twice. Seven reads span four families.
    family = np.arange(7)//2
    for regime in ('independent families','common corruption'):
        errors = rng.random((len(truth),4 if regime=='independent families' else 1)) < .15
        reports = truth[:,None] ^ np.broadcast_to(errors,(len(truth),4))[:,family]
        for mode in ('pooled','family'):
            evidence = np.zeros(len(truth),int)
            active = np.ones(len(truth),bool)
            wrong = reads = 0
            for i in range(7):
                reads += int(active.sum())
                if mode=='family' and i and family[i]==family[i-1]:
                    continue
                evidence += (reports[:,i]*2-1)*active
                promote = active & (abs(evidence)*np.log(.85/.15)>=np.log(499))
                wrong += int(((evidence>0)!=truth)[promote].sum())
                active[promote] = False
            rows.append(dict(seed=seed,regime=regime,mode=mode,n=len(truth),
                             promoted=int((~active).sum()),wrong=wrong,reads=reads))
    return rows


if __name__=='__main__':
    for name,fn in [('index',memory),('promotion',verification)]:
        rows = [r for seed in range(301,306) for r in fn(seed)]
        Path(name+'.json').write_text(json.dumps(rows,indent=2))
