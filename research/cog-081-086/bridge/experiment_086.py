"""COG-ATTN-086a controlled puzzle transfer of the *real COG-066 option acquirer*.

The COG source's acquire() is used without modification, with certified
training-only action traces. A new generic, exact A* planner accepts its learned
options on maze/sliding/Hanoi environments; no neural COG controller is present.

This isolates the option mechanism's causal effect while avoiding a false claim
that COG-ATTN has independently learned puzzle reasoning.
"""
from __future__ import annotations
import argparse, hashlib, heapq, json, random, statistics, sys
from collections import defaultdict, Counter
from pathlib import Path
from time import perf_counter_ns
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_submissions import from_case
from puzzles import Maze, Sliding, Hanoi
from bridge.cog066_source import acquire

DOMAINS = ('maze_7','maze_11','maze_shift','slide_8','slide_14','hanoi_3','hanoi_5')
VALID_ACTIONS = ('U','D','L','R')


def task_state(p):
    if isinstance(p,Maze):return p.start
    return p.start


def goal(p,s):
    if isinstance(p,Maze):return s==p.target
    if isinstance(p,Sliding):return s==p.target
    return s==p.target


def heuristic(p,s):
    if isinstance(p,Maze):return abs(s[0]-p.target[0])+abs(s[1]-p.target[1])
    if isinstance(p,Sliding):return p.heuristic(s)
    return 0


def exact_search(p, options=(), max_pops=30000):
    """Matched search, differing only by availability of verified option edges.

    A candidate shortcut is never trusted. It is replayed one primitive at a
    time against current state and rejected without mutation upon illegality.
    """
    stamp=0;start=task_state(p);best={start:0};prior={start:None};edge={}
    queue=[(heuristic(p,start),0,stamp,start)]
    pop=primitive_attempts=option_attempts=valid_options=invalid_options=0
    began=perf_counter_ns()
    while queue:
        _, g, _, state=heapq.heappop(queue)
        if g!=best[state]:continue
        if pop>=max_pops:break
        pop+=1
        if goal(p,state):
            out=[];k=state
            while prior[k] is not None:
                out.extend(reversed(edge[k]));k=prior[k]
            out.reverse()
            return dict(solved=bool(p.apply(out)),path=out,popped=pop,primitive_attempts=primitive_attempts,
                        option_attempts=option_attempts,valid_options=valid_options,invalid_options=invalid_options,
                        elapsed_ns=perf_counter_ns()-began)
        neighbors=dict(p.neighbors(state))
        # Options first as upstream COG-066 planner; single-step primitives retained.
        for seq in (*options,*((a,) for a in neighbors)):
            cur=state
            is_option=len(seq)>1
            if is_option:option_attempts+=1
            valid=True
            for a in seq:
                primitive_attempts+=1
                nxt=dict(p.neighbors(cur)).get(a)
                if nxt is None:
                    valid=False;break
                cur=nxt
            if is_option:
                valid_options+=int(valid)
                invalid_options+=int(not valid)
            if valid:
                ng=g+len(seq)
                if ng<best.get(cur,10**10):
                    best[cur]=ng;prior[cur]=state;edge[cur]=seq
                    stamp+=1
                    heapq.heappush(queue,(ng+heuristic(p,cur),ng,stamp,cur))
    return dict(solved=False,path=[],popped=pop,primitive_attempts=primitive_attempts,
                option_attempts=option_attempts,valid_options=valid_options,invalid_options=invalid_options,
                elapsed_ns=perf_counter_ns()-began)


def ref_path(p):
    if isinstance(p,(Maze,Sliding,Hanoi)):
        return p.shortest().path
    raise TypeError(type(p))


def learn(cases, train_seeds=(8101,8107)):
    train=defaultdict(list);certified=Counter();t=perf_counter_ns()
    for r in cases:
        if r['split']!='primary' or r['domain'] not in DOMAINS or int(r['id'].split('/')[1]) not in train_seeds:continue
        p=from_case(r);path=ref_path(p)
        valid=bool(path) and p.apply(path)
        assert valid,(r['id'],'reference is not valid')
        # Different family ID only when the actual puzzle state differs.
        train[r['domain']].append(dict(verified=valid,path=path,source_family=r['environment_digest']))
        certified[r['domain']]+=1
    options={k:acquire(train[k]) for k in DOMAINS}
    return options,dict(certified=dict(certified),elapsed_ns=perf_counter_ns()-t)


def scramble_options(options,seed=886):
    """Quantity- and length-matched negative control; not necessarily inert."""
    rng=random.Random(seed);out={}
    for d,opts in options.items():
        pool=[x for seq in opts for x in seq]
        rng.shuffle(pool)
        offset=0;changed=[]
        for seq in opts:
            n=len(seq);changed.append(tuple(pool[offset:offset+n]));offset+=n
        out[d]=tuple(changed)
    return out


def full_run(cases_path=ROOT/'puzzle_cases.jsonl',out_path=ROOT/'bridge'/'086-results.json'):
    cases=[json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    assert len(cases)==396
    assert len(set(r['id'] for r in cases))==len(cases)
    # Source-domain controls leave the primary partition for acquisition only.
    learned,setup=learn(cases)
    shuffled=scramble_options(learned)
    # Separate development seed. Do not use replication to admit options.
    gate={};calibration={};gate_start=perf_counter_ns()
    for domain in DOMAINS:
        dev=[c for c in cases if c['domain']==domain and c['split']=='primary' and c['id'].split('/')[1]=='8113']
        assert len(dev)==6
        dev_rows=[]
        for c in dev:
            p=from_case(c)
            a=exact_search(p)
            b=exact_search(p,learned[domain])
            assert a['solved'] and b['solved'] and len(a['path'])==len(b['path'])
            dev_rows.append((a,b))
        baseline_attempts=sum(x['primitive_attempts'] for x,y in dev_rows)
        candidate_attempts=sum(y['primitive_attempts'] for x,y in dev_rows)
        baseline_ns=sum(x['elapsed_ns'] for x,y in dev_rows)
        candidate_ns=sum(y['elapsed_ns'] for x,y in dev_rows)
        # Strict joint economic admission, including candidate validation.
        admitted=bool(learned[domain]) and candidate_attempts<baseline_attempts and candidate_ns<baseline_ns
        gate[domain]=admitted
        calibration[domain]=dict(n=len(dev_rows),admitted=admitted,primitive_attempts=baseline_attempts,
                                learned_attempts=candidate_attempts,primitive_ns=baseline_ns,learned_ns=candidate_ns)
    gate_ns=perf_counter_ns()-gate_start
    rows=[]
    for case in sorted((c for c in cases if c['split']=='replication' and c['domain'] in DOMAINS),key=lambda c:c['id']):
        p=from_case(case)
        arms=['primitive','cog066','shuffled','gated']
        random.Random(int(hashlib.sha256(case['id'].encode()).hexdigest()[:12],16)).shuffle(arms)
        by={}
        for arm in arms:
            opts=() if arm=='primitive' or (arm=='gated' and not gate[case['domain']]) else learned[case['domain']] if arm in ('cog066','gated') else shuffled[case['domain']]
            result=exact_search(p,opts)
            assert result['solved'],(case['id'],arm,'search failure')
            by[arm]=result
        assert all(len(z['path'])==len(by['primitive']['path']) for z in by.values()),(case['id'],'unequal path length')
        for arm,result in by.items():
            rows.append({**{k:case[k] for k in ('id','domain','split')},'seed':int(case['id'].split('/')[1]),
                         'arm':arm,'option_count':len(opts),
                         **{k:v for k,v in result.items() if k!='path'},'steps':len(result['path']),
                         'path_sha256':hashlib.sha256(repr(result['path']).encode()).hexdigest()})
    by_arm={}
    for arm in ('primitive','cog066','shuffled','gated'):
        r=[x for x in rows if x['arm']==arm]
        by_arm[arm]=dict(n=len(r),solved=sum(x['solved'] for x in r),attempts=sum(x['primitive_attempts'] for x in r),
                         popped=sum(x['popped'] for x in r),macro_probes=sum(x['option_attempts'] for x in r),
                         valid_macro_probes=sum(x['valid_options'] for x in r),
                         median_ms=round(statistics.median(x['elapsed_ns'] for x in r)/1e6,5),
                         total_ms=round(sum(x['elapsed_ns'] for x in r)/1e6,4))
    paired={}
    for domain in DOMAINS:
        rr={arm:[r for r in rows if r['arm']==arm and r['domain']==domain] for arm in ('primitive','cog066','shuffled','gated')}
        paired[domain]={'n':len(rr['primitive']),'options':len(learned[domain]),
                        'primitive_attempts':sum(r['primitive_attempts'] for r in rr['primitive']),
                        'cog_attempts':sum(r['primitive_attempts'] for r in rr['cog066']),
                        'shuffled_attempts':sum(r['primitive_attempts'] for r in rr['shuffled']),
                        'primitive_pop':sum(r['popped'] for r in rr['primitive']),
                        'cog_pop':sum(r['popped'] for r in rr['cog066']),
                        'shuffled_pop':sum(r['popped'] for r in rr['shuffled']),
                        'gated_attempts':sum(r['primitive_attempts'] for r in rr['gated']),
                        'gated_pop':sum(r['popped'] for r in rr['gated'])}
    report={'name':'COG-ATTN-086a option-subsystem supervised transfer (not full COG agent)',
            'source':'COG-066 acquire() copied verbatim from commit 975732f1081d4e5c98c460df1cfa067c48af213d',
            'limitations':['training uses reference solver demonstrations','eval search is exact A* with optional macros, not neural policy','no skill induction from sparse rewards','canonical Hanoi repeats and not independent'],
            'acquisition':setup,'admission_profile_ns':gate_ns,'calibration':calibration,'options':{k:[list(s) for s in v] for k,v in learned.items()},
            'totals':by_arm,'by_domain':paired,'rows':rows}
    Path(out_path).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default=str(ROOT/'puzzle_cases.jsonl'))
    ap.add_argument('--out',default=str(ROOT/'bridge'/'086-results.json'));args=ap.parse_args()
    report=full_run(args.cases,args.out)
    print(json.dumps({k:report[k] for k in ('acquisition','totals','by_domain')},indent=2))
