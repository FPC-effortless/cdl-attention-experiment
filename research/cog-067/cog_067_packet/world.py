"""Synthetic event queues with procedural, separately implemented observations."""
from collections import deque
from hashlib import sha256
from random import Random

PANELS={'train':(81001,240),'calibrate':(81002,80),'dev':(81003,160)}
TASKS=12
ACQ_SLOTS=8
DEADLINE=12
COST={'direct':1,'replay':2,'mirror':1}
HORIZON=12
HISTORY=8
EFFECTS=('strong','weak','null','reverse')
FAULTS=('clean','snapshot_stale','journal_missing','exporter_flip')

def cases(panel):
    seed,n=PANELS[panel];rng=Random(seed)
    out=[]
    for group in range(n):
        effect=EFFECTS[group%len(EFFECTS)]
        fault=FAULTS[(group//len(EFFECTS))%len(FAULTS)]
        rows=[]
        for task in range(TASKS):
            rate=[rng.randrange(1,4) for _ in range(2)]
            workload=[[max(0,x+rng.choice((-2,-1,0,1))) for _ in range(HORIZON)] for x in rate]
            # Conditionally related through latent rates, never reused future values.
            cap=[rng.choice((1,2)) for _ in range(2)]
            early=[[max(0,x+rng.choice((-2,-1,0,1))) for _ in range(HISTORY)] for x in rate]
            focus=rng.randrange(2)
            rows.append({'workload':workload,'early':early,'caps':cap,'focus':focus,
                        'effect':effect,'fault':fault,'salt':rng.randrange(4)})
        out.append((f'{panel}-{group:04}',rows))
    return out

def work(a,caps,carry=(0,0),delta=(0,0)):
    """Direct worker's FIFO counters; queue occupancy, discharged items, snapshot."""
    qs=[deque([0]*x) for x in carry]; finished=[0,0]; old=(0,0)
    for index,(n0,n1) in enumerate(zip(*a)):
        old=tuple(map(len,qs))
        for j,n in enumerate((n0,n1)):
            qs[j].extend([index]*n)
            for _ in range(min(len(qs[j]),max(0,caps[j]+delta[j]))):qs[j].popleft();finished[j]+=1
    return tuple(map(len,qs)),tuple(finished),old

def replay(case, missing=True):
    """Independent implementation: weighted ingress recurrence, not exporter output."""
    if missing and case['fault']=='journal_missing':return None
    # Replay reconstructs source-owned arrival totals by decoding a journal.
    # It yields utilization trend (not the worker's queue occupancy statistic).
    events=bytes(v for row in zip(*case['early']) for v in row)
    net=[0,0]
    for n,v in enumerate(events):net[n%2]+=v-case['caps'][n%2]
    return direction(net)

def direction(q):
    d=q[0]-q[1]
    return -1 if d<=-2 else 1 if d>=2 else 0

def direct(case):
    q,_,old=work(case['early'],case['caps'])
    return direction(old if case['fault']=='snapshot_stale' else q)

def passive(case):
    # Cheap existing exporter. Source lineage is distinct from replay, not supply chain independence.
    early=case['early']
    q=[sum(stream[-2:]) for stream in early]
    ans=direction(q)
    if case['fault']=='exporter_flip':ans=-ans
    return ans

def observe(case,source):
    if source=='direct': return direct(case)
    if source=='replay': return replay(case)
    if source=='mirror': return passive(case)
    raise ValueError(source)

def truth(case):
    carry,_,_=work(case['early'],case['caps'])
    # Use identical future arrivals in paired clones; active action costs one utility unit.
    diff={'strong':2,'weak':1,'null':0,'reverse':-1}[case['effect']]
    reward=[]
    for action in (0,1,2):
        bump=tuple(diff if action==j+1 else 0 for j in range(2))
        _,done,_=work(case['workload'],case['caps'],carry,bump)
        reward.append(sum(done)-(action!=0))
    return reward

def stratum(case):
    return case['effect'],case['fault']
