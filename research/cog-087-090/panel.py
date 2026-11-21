"""Locked held/rep panels and an independent shortest-distance evaluator."""
from collections import deque
from dataclasses import replace
from hashlib import sha256
import json,os,random,sys
from pathlib import Path
from time import perf_counter_ns
from tasks import tasks,DOMAINS
from search import solve

ROOT=Path(__file__).resolve().parent
def inputs(part,model):
    training=[*tasks('train'),*tasks('dev')]
    oldgoals={(t.kind,t.target) for t in training if t.kind!='maze'}
    seen=set(model['train_digests']+model['dev_digests'])
    for split in ('held','rep'):
        panel=[]
        for t in tasks(split):
            rng=random.Random(t.id+'/deduplicate')
            while t.digest() in seen or (t.kind,t.target) in oldgoals or t.start==t.target:
                t=replace(t,target=rng.choice(t.neighbors(t.target))[1])
            seen.add(t.digest());panel.append(t)
        if part==split:return panel
    raise ValueError(part)

def edges(t,s):
    if t.kind=='hanoi':
        for d,src in enumerate(s):
            if src in s[:d]:continue
            for target in range(t.pegs):
                if (src,target) in t.links and target not in s[:d]:
                    yield (src,target),(*s[:d],target,*s[d+1:])
        return
    p=s if t.kind=='maze' else divmod(s.index(0),3)
    for key,dd in (('R',(0,1)),('L',(0,-1)),('D',(1,0)),('U',(-1,0))):
        y,x=p[0]+dd[0],p[1]+dd[1]
        if t.kind=='maze':
            if (y,x) in t.cells:yield key,(y,x)
        elif 0<=x<3 and 0<=y<3:
            i=s.index(0);j=3*y+x;b=list(s);b[i],b[j]=b[j],b[i]
            yield key,tuple(b)

def distance(t):
    queue=deque([(t.start,0)]);seen={t.start}
    while queue:
        s,d=queue.popleft()
        if s==t.target:return d
        for _,n in edges(t,s):
            if n not in seen:seen.add(n);queue.append((n,d+1))
    raise ValueError('unreachable')

def options(model,arm,kind):
    if arm=='C':opts=model['shuffled'][kind]
    elif arm=='B' or (arm=='D' and model['gates'][kind]):opts=model['options'][kind]
    else:return ()
    return tuple(tuple(tuple(a) if isinstance(a,list) else a for a in seq) for seq in opts)

def run(part):
    model_path=Path(os.getenv('SKILL_MODEL',ROOT/'results/model.json'))
    model=json.loads(model_path.read_bytes());data=[]
    for t in inputs(part,model):
        arms=['A','B','C','D','E','structural','efficient']
        rng=random.Random(t.id);rng.shuffle(arms);length=distance(t)
        by={}
        for arm in arms:
            # Three measured passes with interleaved arm order below.
            by[arm]=None
        for repeat in range(3):
            rng.shuffle(arms)
            for arm in arms:
                r=solve(t,arm,options(model,arm,t.kind),model,model['tau'][t.kind])
                assert r['steps']==length,(t.id,arm,r['steps'],length)
                s=t.start
                for a in r['actions']:s=dict(edges(t,s))[a]
                assert s==t.target
                if repeat==0:by[arm]=r;by[arm]['times']=[r['ns']]
                else:
                    assert by[arm]['actions']==r['actions']
                    by[arm]['times'].append(r['ns'])
        for arm in by:by[arm]['ns']=sorted(by[arm]['times'])[1]
        data.append(dict(id=t.id,kind=t.kind,fixture=t.digest(),target=t.target,
                         pegs=t.pegs,reference_steps=length,arms=by))
    return dict(panel=part,model_sha256=sha256(model_path.read_bytes()).hexdigest(),tasks=data)

if __name__=='__main__':
    part=sys.argv[1];data=run(part)
    out=Path(os.getenv('TRAINLAB_OUTPUT_DIR',ROOT/'results'))
    (out/(part+'.json')).write_text(json.dumps(data,separators=(',',':'))+'\n')
    (out/'metrics.json').write_text(json.dumps({'solved':len(data['tasks'])}))
