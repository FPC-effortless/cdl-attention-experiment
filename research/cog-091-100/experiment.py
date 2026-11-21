"""COG-ATTN-091/094/096: bounded attention-function substitutes (synthetic diagnostics).

Run: python experiment.py --out results.json
Only Python standard library is needed for 091 and 096. No Transformer or
complete COG-ATTN agent is trained or tested. All results are diagnostic.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import random
import statistics
from time import perf_counter_ns

SEEDS = (9107, 9203, 9301)


def make_history(n: int, seed: int):
    assert n % 3 == 0
    rng = random.Random(seed)
    words = ('cedar', 'harbor', 'olive', 'amber', 'silver', 'willow', 'stone')
    records = []
    groups = list(range(n // 3))
    rng.shuffle(groups)
    for i in range(n // 3):
        entity = f'item{i:06d}'
        area = f'area{rng.randint(0, 40):02d}'
        for j in range(3):
            marker = f'novel{seed}x{i}x{j}'
            day = f'day{rng.randint(1, 30):02d}'
            phrase = f'please reserve {entity} at {area} for {day} with note {marker} {words[j + i % 4]}'
            records.append(dict(entity=entity, marker=marker, day=day, raw=phrase))
    # Ingest a scrambled stream; the latest slot is whichever arrives last.
    rng.shuffle(records)
    return records


def index_raw(records, token_sets=None):
    raw_index = defaultdict(set)
    start = perf_counter_ns()
    for i, row in enumerate(records):
        for word in (token_sets[i] if token_sets is not None else set(row['raw'].split())):
            raw_index[word].add(i)
    return raw_index, perf_counter_ns() - start


def query_index(records, ix, entity, marker):
    candidates = ix.get(entity, set()) & ix.get(marker, set())
    # Fail closed if missing or ambiguous; never guess.
    return records[next(iter(candidates))]['day'] if len(candidates) == 1 else None


def query_scan(records, entity, marker):
    candidates = [row['day'] for row in records if entity in row['raw'].split() and marker in row['raw'].split()]
    return candidates[0] if len(candidates) == 1 else None


def memory_panel(n: int, seed: int, q_count: int = 128):
    rows = make_history(n, seed)
    eager = {row['entity']: row['day'] for row in rows}
    prep_start = perf_counter_ns()
    token_sets=[set(row['raw'].split()) for row in rows]
    tokenize_ns=perf_counter_ns()-prep_start
    ix, build_ns = index_raw(rows,token_sets)
    q_rng = random.Random(seed ^ 0x9988)
    queries = q_rng.sample(rows, q_count)
    outcomes = {name: dict(correct=0, lookup_ns=0) for name in ('eager','raw_scan','cached_scan','lossless_index')}
    for row in queries:
        e,t,g = row['entity'], row['marker'], row['day']
        for name,run in [('eager',lambda:eager.get(e)),('raw_scan',lambda:query_scan(rows,e,t)),
                         ('cached_scan',lambda:next((rows[i]['day'] for i,tokens in enumerate(token_sets) if e in tokens and t in tokens),None)),
                         ('lossless_index',lambda:query_index(rows,ix,e,t))]:
            start = perf_counter_ns(); answer = run()
            outcomes[name]['lookup_ns'] += perf_counter_ns() - start
            outcomes[name]['correct'] += (answer == g)
    # Conservative accounting: index posts for every *distinct* word, not keyed slots.
    return dict(n=n,seed=seed,queries=q_count,build_ns=build_ns,tokenize_ns=tokenize_ns,
                raw_bytes=sum(len(x['raw'].encode()) for x in rows),
                index_postings=sum(len(x) for x in ix.values()),
                outcomes={name:{**v,'accuracy':v['correct']/q_count,
                                 'average_lookup_us':v['lookup_ns']/q_count/1000} for name,v in outcomes.items()})


# A limited search language. It can infer a new operator LABEL from demonstrations,
# but cannot invent entirely new computational primitives or a new DSL.
def reverse(x):return x[::-1]
def rotate_l(x):return x[1:]+x[:1]
def rotate_r(x):return x[-1:]+x[:-1]
def swap_first(x):return x[1:2]+x[:1]+x[2:]
def duplicate_last(x):return x+x[-1:]
OPS = dict(reverse=reverse,rotate_l=rotate_l,rotate_r=rotate_r,
           swap_first=swap_first,duplicate_last=duplicate_last)
PROGRAMS = tuple((p,) for p in OPS) + tuple((p,q) for p in OPS for q in OPS)

def execute(program,x):
    for op in program:x=OPS[op](x)
    return x


def make_string(rng, length=6):
    # Unique characters avoid accidental collisions in distinguishing examples.
    return ''.join(rng.sample('abcdefghijklmnopqrstuvwx',length))


def find_program(witnesses):
    matching = [p for p in PROGRAMS if all(execute(p,x)==y for x,y in witnesses)]
    # Observational equivalence does not license a guess about program identity.
    return matching[0] if len(matching)==1 else None, len(matching)


def skill_panel(seed: int, episodes: int = 90):
    rng=random.Random(seed)
    correct = admitted = ambiguous = impossible = wrong = 0
    cached_correct=0
    no_commit_on_bad=0
    cost_tests=0
    per_case=[]
    for ep in range(episodes):
        true_program=rng.choice(PROGRAMS)
        label=f'novelexpr{seed}_{ep}' # no pre-programmed operator-name binding
        seen={}
        for _ in range(5):
            x=make_string(rng)
            seen[x]=execute(true_program,x)
        query=make_string(rng)
        while query in seen:query=make_string(rng)
        corrupt=(ep%10==0)
        witnesses=list(seen.items())
        if corrupt:
            # Inconsistent supplied observation: no allowed program produces it.
            x,y=witnesses[-1];witnesses[-1]=(x,'#'+y)
        before=perf_counter_ns()
        program,match_count=find_program(witnesses)
        latency=perf_counter_ns()-before
        cost_tests+=len(PROGRAMS)*len(witnesses)
        prediction=execute(program,query) if program is not None else None
        gold=execute(true_program,query)
        success=(prediction == gold) if not corrupt else (prediction is None)
        correct+=success
        admitted+=program is not None
        ambiguous+=match_count>1
        impossible+=match_count==0
        wrong+=(prediction is not None and prediction!=gold)
        no_commit_on_bad+=int(corrupt and prediction is None)
        cached_correct+=int(query in seen and seen[query]==gold)
        per_case.append(dict(ep=ep,corrupt=corrupt,program=list(true_program),
                             recognized=list(program) if program else None,
                             candidates=match_count,correct=bool(success),
                             latency_ns=latency,label_sha256=hashlib.sha256(label.encode()).hexdigest()))
    return dict(seed=seed,n=episodes,correct=correct,admitted=admitted,
                ambiguous=ambiguous,contradictory=impossible,wrong_actions=wrong,
                corrupt_abstentions=no_commit_on_bad,
                exact_demo_cache_correct=cached_correct,checks_upper_bound=cost_tests,
                average_induce_us=sum(z['latency_ns'] for z in per_case)/episodes/1000,
                cases=per_case)


def run():
    memory=[memory_panel(n,s) for n in (1536,6144,24576) for s in SEEDS]
    skill=[skill_panel(s) for s in SEEDS]
    out=dict(program='COG-ATTN-091 and 096 pilot (synthetic, not complete agent)',
             seeds=list(SEEDS), memory=memory, skill=skill,
             limitations=['Raw-index retrieval requires exact lexeme overlap, and consumes growing storage',
                          'Raw-scan is a dense-lookup control, NOT a trained self-attention Transformer',
                          'The program inducer uses a supplied five-operation DSL, not zero-schema intelligence',
                          'All data are synthetic; no authentic COG-ATTN controller or learned language interface',
                          'Timings are local CPU diagnostics, not hardware-normalized or end-to-end costs'])
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results.json')
    args=p.parse_args()
    r=run()
    with open(args.out,'w') as f:json.dump(r,f,indent=2,sort_keys=True)
    for n in (1536,6144,24576):
        rows=[x for x in r['memory'] if x['n']==n]
        print('memory',n, {k: {'accuracy':round(statistics.mean(x['outcomes'][k]['accuracy'] for x in rows),4),
                                  'lookup_us':round(statistics.median(x['outcomes'][k]['average_lookup_us'] for x in rows),2)}
                           for k in ('eager','raw_scan','cached_scan','lossless_index')})
    print('induction', {'correct':sum(x['correct'] for x in r['skill']),
                       'episodes':sum(x['n'] for x in r['skill']),
                       'admitted':sum(x['admitted'] for x in r['skill']),
                       'wrong_actions':sum(x['wrong_actions'] for x in r['skill']),
                       'baseline_demo_cache_correct':sum(x['exact_demo_cache_correct'] for x in r['skill'])})
