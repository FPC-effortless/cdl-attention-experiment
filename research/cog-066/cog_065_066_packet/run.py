"""Closed phases: preflight/seal, acquire, predict, then independent verify."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns
import sys
from .episodes import cases,execute
from .planning import acquire

ROOT=Path(__file__).parent
MANIFEST=ROOT/"FREEZE.json"
ARM_NAMES=('primitive','learned','fixed','shuffled_6501','shuffled_6502','shuffled_6503')

def digest(p): return sha256(p.read_bytes()).hexdigest()
def dump(p,v): 
    with p.open('x') as stream: json.dump(v,stream,sort_keys=True,indent=2)
def read(p):return json.loads(p.read_text())
def protocol_files():return sorted([*ROOT.glob('*.py'),*ROOT.glob('*.md')])
def inventory():
    return {p.name:digest(p) for p in protocol_files()}
def seal():
    ds=cases() # Metadata only; no trajectory search or evaluation.
    serial={k:[(s,t,i,asdict(w)) for s,t,i,w in rows] for k,rows in ds.items()}
    p={'sources':inventory(),'python':sys.version,'platform':sys.platform,
       'data_hash':sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest(),
       'case_counts':{k:len(v) for k,v in ds.items()}}
    dump(MANIFEST,p)
    print(p)
def verify_freeze():
    m=read(MANIFEST)
    assert m['sources']==inventory(), 'Frozen scientific source change'
    ds=cases()
    serial={k:[(s,t,i,asdict(w)) for s,t,i,w in rows] for k,rows in ds.items()}
    assert m['data_hash']==sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest()
    return ds

def fixed_options():
    # Uniform recognizable MOVE-THEN-WORK, plus deterministic WORK-THEN-WORK.
    return tuple((bay,4+item) for bay in range(4) for item in range(3))+((4,5),(4,6),(5,4),(6,4))

def shuffle_options(ops,seed):
    rng=Random(seed)
    out=[]
    for seq in ops:
        seq=list(seq);original=tuple(seq)
        for _ in range(16):
            rng.shuffle(seq)
            if tuple(seq)!=original:break
        out.append(tuple(seq))
    return tuple(out)

def acquisition(out,ds):
    start=perf_counter_ns()
    from .verify import observe,verify_acquisition
    traces=[]
    for sig,variant,focus,w in ds['train']:
        if variant!='success':continue
        r=execute(w,focus)
        v=observe(w,r)
        assert v[1]=='completed'
        traces.append({'source_family':sig,'verified':True,'path':[rec[1] for rec in r['receipts']],'execution':r})
    trainconstruct=perf_counter_ns()-start
    start=perf_counter_ns();learned=acquire(traces);learned_cost=perf_counter_ns()-start
    start=perf_counter_ns();fixed=fixed_options();fixed_cost=perf_counter_ns()-start
    costs={'primitive':0,'learned':trainconstruct+learned_cost,'fixed':fixed_cost}
    dictionaries={'primitive':(), 'learned':learned, 'fixed':fixed}
    for seed in (6501,6502,6503):
        label=f'shuffled_{seed}'; start=perf_counter_ns()
        dictionaries[label]=shuffle_options(learned,seed)
        costs[label]=trainconstruct+learned_cost+perf_counter_ns()-start
    audit_start=perf_counter_ns()
    audited=verify_acquisition(ds['train'],traces)
    audit_ns=perf_counter_ns()-audit_start
    dump(out/'TRAIN_TRACES.json',traces)
    dump(out/'ACQUISITION_CLOSED.json',{'source_seal':digest(MANIFEST),'trace_sha256':digest(out/'TRAIN_TRACES.json'),'options':dictionaries,'build_ns':costs,'oracle_audited':audited,'oracle_audit_ns':audit_ns})
    print('ACQUISITION_CLOSED',len(traces),len(learned))

def timing_indices(ds):
    return {p:[i for i,(sig,var,_,_) in enumerate(v)
        if int.from_bytes(sha256(bytes(sig)).digest()[1:3],'big')%4==0]
        for p,v in ds.items() if p!='train'}

def predict(out,ds):
    acq=read(out/'ACQUISITION_CLOSED.json')
    assert acq['source_seal']==digest(MANIFEST)
    # Native dispatcher validates the acquisition trace hash before staging,
    # but deliberately does not pass raw training traces into prediction.
    ops={k:tuple(map(tuple,v)) for k,v in acq['options'].items()}
    panels={k:v for k,v in ds.items() if k!='train'}
    rows=[];timings=[];expected={}
    from .reporting import template
    from .episodes import flat_fields
    from .verify import observe
    def timed_work(case,arm):
        sig,var,focus,w=case
        r=execute(w,focus,ops[arm])
        observe(w,r)
        return r,template(flat_fields(w,r))
    indices=timing_indices(ds)
    for panel,caseset in panels.items():
        for i,case in enumerate(caseset):
            sig,var,focus,w=case
            for arm in ARM_NAMES:
                r,t=timed_work(case,arm)
                rows.append((panel,i,arm,r,t))
                expected[panel,i,arm]=(r,t)
        for i in indices[panel][:6]:
            for arm in ARM_NAMES: timed_work(caseset[i],arm)
        for repeat in range(7):
            keys=[(i,a) for i in indices[panel] for a in ARM_NAMES]
            Random(66000+repeat).shuffle(keys)
            for i,a in keys:
                t=perf_counter_ns();candidate=timed_work(caseset[i],a)
                elapsed=perf_counter_ns()-t
                assert candidate==expected[panel,i,a], 'Timed output differs from frozen prediction'
                timings.append((panel,i,a,repeat,elapsed))
    dump(out/'PREDICTIONS.json',rows);dump(out/'TIMINGS.json',timings)
    dump(out/'PREDICTIONS_CLOSED.json',{n:digest(out/n) for n in ('PREDICTIONS.json','TIMINGS.json','ACQUISITION_CLOSED.json')})
    print('PREDICTIONS_CLOSED',len(rows),len(timings))

def verify(out,ds):
    from .verify import observe,score,exact_cost
    from collections import defaultdict
    acq=read(out/'ACQUISITION_CLOSED.json')
    closed=read(out/'PREDICTIONS_CLOSED.json')
    assert all(digest(out/n)==h for n,h in closed.items())
    by=defaultdict(dict); samples=defaultdict(list);oracle_costs={}
    for panel,i,a,r,t in read(out/'PREDICTIONS.json'):
        sig,var,focus,w=ds[panel][i]
        assert focus==r['focus'] and a not in by[panel,i]
        truth=observe(w,r)
        assert score(t,truth)['strict']
        if var=='success':
            if (panel,i) not in oracle_costs:oracle_costs[panel,i]=exact_cost(w)
            if a=='primitive':assert truth[1]=='completed' and r['spent']==oracle_costs[panel,i]
        by[panel,i][a]=r
    seen_timing=set()
    for p,i,a,rep,ns in read(out/'TIMINGS.json'):
        key=(p,i,a,rep)
        assert key not in seen_timing and ns>0
        seen_timing.add(key)
        samples[p,i,a].append(ns)
    assert seen_timing=={(p,i,a,rep) for p,is_ in timing_indices(ds).items()
                         for i in is_ for a in ARM_NAMES for rep in range(7)}
    expected={(p,i) for p in ds if p!='train' for i in range(len(ds[p]))}
    assert set(by)==expected and all(set(v)==set(ARM_NAMES) for v in by.values())
    pairs=defaultdict(list);group_cases=defaultdict(list)
    for p,i in expected:group_cases[p,ds[p][i][0]].append(i)
    for (p,i,a),ns in samples.items():
        assert len(ns)==7
        pairs[p,ds[p][i][0],a].append(median(ns))
    assert all(len(v)==3 for v in pairs.values())
    stats={}
    for a in ARM_NAMES:
        bad=sum(v[a]['status']!=v['primitive']['status'] or
          (v['primitive']['status']=='completed' and v[a]['spent']>v['primitive']['spent']) for v in by.values())
        success=sum(v[a]['status']=='completed' for v in by.values())
        stage={k:sum(v[a][k] for v in by.values()) for k in ('expanded','transition_attempts','option_attempts','planned_options','finished_options','spent')}
        families=sorted((p,sig) for p,sig,x in pairs if x==a)
        net=[(sum(pairs[p,sig,'primitive'])-sum(pairs[p,sig,a]))/3 for p,sig in families]
        speed=[1-sum(pairs[p,sig,a])/sum(pairs[p,sig,'primitive']) for p,sig in families]
        mean=sum(net)/len(net)
        rng=Random(66006);reps=sorted(sum(rng.choices(net,k=len(net)))/len(net) for _ in range(2000))
        build=acq['build_ns'][a]
        work={str(n):build-12*n*mean for n in (512,4096)}
        stats[a]={'quality_regressions':bad,'verified_completions':success,'timed_families':len(families),
            'mean_saved_ns_per_workflow':mean,'median_per_family_fraction_saved':median(speed),'paired_mean_saved_ns_95pct':[reps[50],reps[1949]],'weighted_lifecycle_delta_ns':work,'stages':stage,
            'option_count':len(acq['options'][a]),
            'unique_option_count':len(set(map(tuple,acq['options'][a]))),
            'panels':{p:{'completed':sum(v[a]['status']=='completed' for (p0,_),v in by.items() if p0==p),
                          'median_family_fraction_saved':median(f for f,(p0,_) in zip(speed,families) if p0==p)}
                      for p in ('dev_composition','dev_binding')}}
    a=stats['learned']; f=stats['fixed']
    gate=(a['quality_regressions']==0 and a['median_per_family_fraction_saved']>=.10 and
          a['paired_mean_saved_ns_95pct'][0]>0 and
          a['mean_saved_ns_per_workflow']>=f['mean_saved_ns_per_workflow']-.02*median(median(v) for (p,i,k),v in samples.items() if k=='primitive') and
          a['weighted_lifecycle_delta_ns']['512']<0)
    dump(out/'VERIFIED.json',{'status':'complete','gate':bool(gate),'arm_stats':stats,'oracle_verified_static_reference_cases':len(oracle_costs)})
    print('VERIFIED gate',gate)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('seal',))
    p.add_argument('--output',type=Path,default=ROOT/'run')
    p.add_argument('--review-sha256')
    a=p.parse_args()
    if a.phase=='seal':seal();return
    raise AssertionError('Use native dispatcher for experiments')

if __name__=='__main__':main()

