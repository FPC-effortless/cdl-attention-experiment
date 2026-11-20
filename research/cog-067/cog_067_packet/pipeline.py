"""Separate local fit/calibrate, held prediction, and verifier stages."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
from random import Random
from statistics import mean,median
from time import perf_counter_ns
from . import world as w
from . import policies as p

def scoped(cases):
    """Information permitted at prediction. Deliberately excludes future arrivals/effect."""
    return [(group,[{k:case[k] for k in ('early','caps','fault')} for case in rows]) for group,rows in cases]

def train(cases,cal):
    all_rows=[]
    for _,rows in cases:
        for case in rows:
            all_rows.append((w.passive(case),{s:w.observe(case,s) for s in p.SOURCES},w.truth(case)))
    model=p.fit(all_rows)
    calib=[(w.passive(c),{s:w.observe(c,s) for s in p.SOURCES},w.truth(c)) for _,rs in cal for c in rs]
    def outcomes(x,t):
        pp,obs,r=x
        cells=[]
        for s in (None,'direct','replay','mirror'):
            a,conf,_=p.decide(model,pp,s,obs.get(s),t)
            cells.append((a!='UNKNOWN',a!='UNKNOWN' and max(r)==r[a]))
        return cells
    feasible=[]
    for threshold in (.35,.45,.55,.65,.75,.85,.95):
        o=[v for x in calib for v in outcomes(x,threshold)]
        a=sum(u for u,_ in o);ok=sum(v for _,v in o)
        if a and ok/a>=.80 and a/len(o)>=.55:
            feasible.append((a,threshold))
    # Infeasible calibration is prospectively retained and must fail promotion.
    model['calibration_met']=bool(feasible)
    model['tau']=max(feasible)[1] if feasible else .55
    model['ask_threshold']=.8
    model['ask_floor']=.06
    # Strong nonadaptive action order uses TRAIN expected incremental values by ordinal.
    means={s:[mean(p.evsi(model,w.passive(rows[i]),s) for _,rows in cases) for i in range(w.ACQ_SLOTS)]
           for s in ('direct','replay')}
    schedule=[]
    for r in range(w.ACQ_SLOTS):
        for a,b in combinations([i for i in range(w.ACQ_SLOTS) if i!=r],2):
            schedule.append((means['replay'][r]+means['direct'][a]+means['direct'][b],r,a,b))
    _,r,a,b=max(schedule)
    model['fixed']={str(r):'replay',str(a):'direct',str(b):'direct'}
    # Independent uncertainty-first policy calibrates its priority threshold on CAL only.
    trials=[]
    for bound in (.50,.60,.70,.80,.90):
        temp={**model,'ask_threshold':bound}; regret=0
        for group,rows in cal:
            log=one_group(temp,group,scoped([(group,rows)])[0][1],'learned_uncertainty')
            for case,reported in zip(rows,log):
                r=w.truth(case);chosen=reported.get('decision','UNKNOWN')
                regret+=max(r)-r[0 if chosen=='UNKNOWN' else chosen]
        trials.append((regret,bound))
    _,model['ask_threshold']=min(trials)
    return model

def one_group(model,group,rows,arm):
    current=0;out=[]
    if arm in p.MATCHED:q=p.QUOTA.copy()
    elif arm=='always_ask':q={'direct':w.DEADLINE,'replay':w.DEADLINE}
    else:q={}
    rng=Random(81819+sum((i+1)*ord(c) for i,c in enumerate(group)))
    rnd=rng.sample(list(range(w.ACQ_SLOTS)),3)
    fixed={int(k):v for k,v in model['fixed'].items()}
    opts=({rnd[0]:'replay',rnd[1]:'direct',rnd[2]:'direct'} if arm=='random_matched' else fixed)
    for i,case in enumerate(rows):
        if current>=w.DEADLINE:
            out.append({'slot':i,'status':'DISPLACED','clock':current})
            continue
        pp=w.passive(case)
        s=p.query(model,pp,arm,i,q,w.DEADLINE-current-1,opts)
        v=None;cost=0;ns=0
        if s is not None:
            assert q[s]>0
            cost=w.COST[s]
            if current+1+cost>w.DEADLINE:s=None;cost=0
            else:
                t=perf_counter_ns();v=w.observe(case,s);ns=perf_counter_ns()-t
                q[s]-=1
        current+=1+cost
        decision,conf,proposal=p.decide(model,pp,None if arm=='query_ignore' else s,v,model['tau'])
        out.append({'slot':i,'status':'DECIDED','clock':current,'source':s,'value':v,
                    'source_ns':ns,'decision':decision,'confidence':conf,'proposal':proposal,'passive':pp})
    if arm in p.MATCHED:assert all(v==0 for v in q.values()) and current==w.DEADLINE
    return out

TIMING_REPEATS=5

def invariant(records):
    return [{k:v for k,v in r.items() if k!='source_ns'} for r in records]

def predict(model,cases):
    predictions=[];timing=[]
    for gi,(group,rows) in enumerate(cases):
        baseline={a:one_group(model,group,rows,a) for a in p.ARMS}
        predictions.extend([group,a,baseline[a]] for a in p.ARMS)
        for repetition in range(TIMING_REPEATS):
            order=list(p.ARMS);Random(81200+TIMING_REPEATS*gi+repetition).shuffle(order)
            for arm in order:
                start=perf_counter_ns();records=one_group(model,group,rows,arm);elapsed=perf_counter_ns()-start
                assert invariant(records)==invariant(baseline[arm])
                timing.append([group,arm,repetition,elapsed])
    return predictions,timing

def oracle(case):
    # Independently implemented scalar balance law, no shared plant worker.
    caps=case['caps']; totals=[]
    for act in range(3):
        carry=[0,0]
        for j in (0,1):
            for arrival in case['early'][j]:carry[j]=max(0,carry[j]+arrival-caps[j])
        serviced=0
        effect={'strong':2,'weak':1,'null':0,'reverse':-1}[case['effect']]
        for j in (0,1):
            capacity=max(0,caps[j]+(effect if act==j+1 else 0))
            for arrival in case['workload'][j]:
                old=carry[j]+arrival; carry[j]=max(0,old-capacity)
                serviced+=old-carry[j]
        totals.append(serviced-int(act!=0))
    return totals

def audit(model,case_by_group,preds,times):
    expected={(g,a) for g in case_by_group for a in p.ARMS}
    assert expected=={(g,a) for g,a,_ in preds} and len(preds)==len(expected)
    assert {(g,a,r) for g,a in expected for r in range(TIMING_REPEATS)}=={(g,a,rep) for g,a,rep,_ in times}
    assert len(times)==len(expected)*TIMING_REPEATS and all(t>0 for _,_,_,t in times)
    metrics={a:Counter() for a in p.ARMS};groups=defaultdict(dict)
    strength={a:defaultdict(Counter) for a in p.ARMS}
    fault_tables={a:defaultdict(Counter) for a in p.ARMS}
    joint_tables={a:defaultdict(Counter) for a in p.ARMS}
    oracle_rows=defaultdict(list);offline_conditional=defaultdict(Counter);agreement=defaultdict(Counter)
    # This additional information audit is verifier-only. No full-source data or gold enters prediction.
    for rows in case_by_group.values():
        for c in rows:
            r=oracle(c); pp=w.passive(c); best=max(r); base=r[0]
            for stratum in (c['effect'],f"{c['effect']}:{c['fault']}"):
                oracle_rows[stratum].append((best-base,r[1]-base,r[2]-base))
            vals={s:w.observe(c,s) for s in p.SOURCES}
            a=agreement[c['fault']]
            a['rows']+=1;a['mirror_matches_passive']+=int(vals['mirror']==pp)
            if vals['replay'] is not None:
                a['direct_replay_nonmissing']+=1;a['direct_replay_agree']+=int(vals['direct']==vals['replay'])
            for source in (None,*p.SOURCES):
                proposed,_,_=p.decide(model,pp,source,vals.get(source),model['tau'])
                d=offline_conditional[f"{c['fault']}:{source}"]
                d['rows']+=1;d['known']+=int(proposed!='UNKNOWN')
                d['regret']+=best-r[0 if proposed=='UNKNOWN' else proposed]
    curves={a:[] for a in p.ARMS};clocks={}
    for g,arm,rows in preds:
        assert len(rows)==w.TASKS
        ms=metrics[arm];elapsed=0
        acquired=Counter();regret=0
        for i,(r,c) in enumerate(zip(rows,case_by_group[g])):
            assert r['slot']==i
            reward=oracle(c); base=reward[0]; best=max(reward)
            st=strength[arm][c['effect']];st['scheduled']+=1
            ft=fault_tables[arm][c['fault']];ft['scheduled']+=1
            jt=joint_tables[arm][f"{c['effect']}:{c['fault']}"];jt['scheduled']+=1
            if r['status']=='DISPLACED':
                assert elapsed==w.DEADLINE and r['clock']==elapsed
                ms['displaced']+=1;st['displaced']+=1;ft['displaced']+=1;regret+=best-base
                st['regret']+=best-base;ft['regret']+=best-base;jt['regret']+=best-base;jt['displaced']+=1
                ms['displaced_opportunity_regret']+=best-base
                continue
            s=r['source'];cost=0 if s is None else w.COST[s]
            if s is not None:acquired[s]+=1
            assert r['passive']==w.passive(c)
            assert (r['value'] is None if s is None else r['value']==w.observe(c,s))
            elapsed+=1+cost
            assert r['clock']==elapsed<=w.DEADLINE
            if not r['source_ns'] and s is not None:raise AssertionError('missing acquired-source timing')
            d=r['decision'];assert 0<=r['confidence']<=1 and d in (0,1,2,'UNKNOWN')
            ms['served']+=1;st['served']+=1;ft['served']+=1;jt['served']+=1
            ms['acquisition_ticks']+=cost
            ms['source_wall_ns_total']+=r['source_ns']
            ms['acquired_'+str(s)]+=int(s is not None)
            if s is not None and r['value'] is None:ms['missing_reads']+=1
            picked=base if d=='UNKNOWN' else reward[d]
            regret+=best-picked;st['regret']+=best-picked;ft['regret']+=best-picked;jt['regret']+=best-picked
            if d=='UNKNOWN':ms['unknown']+=1;st['unknown']+=1;ft['unknown']+=1;jt['unknown']+=1
            else:
                ms['correct']+=int(picked==best);ms['decisions']+=1
                if d:
                    ms['active']+=1;ms['correct_active']+=int(picked==best)
                    ms['harmful_interventions']+=int(picked<base)
                if c['effect']=='null' and d:ms['null_interventions']+=1;st['null_interventions']+=1
            # Unknown does not change live policy or falsely imply no effect.
            proposal=r['proposal'];assert proposal in (0,1,2)
            assert (d,r['confidence'],proposal)==p.decide(model,r['passive'],None if arm=='query_ignore' else s,r['value'],model['tau'])
            curves[arm].append((r['confidence'],d,reward[proposal]==best,s,c['effect']))
        if arm in p.MATCHED:
            assert acquired==p.QUOTA
            assert sum(w.COST[s]*n for s,n in acquired.items())==4 and elapsed==w.DEADLINE
            assert Counter(r['status'] for r in rows)=={'DECIDED':8,'DISPLACED':4}
        if arm=='passive':assert not acquired
        clocks[g,arm]=elapsed;groups[g][arm]=regret
    t_groups=defaultdict(list)
    for g,k,_,t in times:t_groups[g,k].append(t)
    times_by={a:[median(t_groups[g,a]) for g in case_by_group] for a in p.ARMS}
    def interval(diffs,seed):
        rng=Random(seed); vals=sorted(mean(rng.choices(diffs,k=len(diffs))) for _ in range(2000))
        return [vals[50],vals[1949]]

    contrasts={}
    for a in p.MATCHED:
        if a=='adaptive':continue
        diffs=[v[a]-v['adaptive'] for v in groups.values()]
        latency=[median(t_groups[g,'adaptive'])-median(t_groups[g,a]) for g in case_by_group]
        contrasts[a]={'regret_reduction':mean(diffs),'ci95':interval(diffs,81111),
                      'extra_session_ns':mean(latency),'extra_session_ns_ci95':interval(latency,81112),
                      'weighted_added_inference_ns_12x':12*mean(latency)}
    result={a:{**metrics[a],'regret':sum(v[a] for v in groups.values()),
                  'median_session_ns':median(times_by[a]),'effect':dict(strength[a]),'fault':dict(fault_tables[a]),'effect_fault':dict(joint_tables[a])}
            for a in p.ARMS}
    for arm,vals in curves.items():
        curve=[]
        for t in (0,.35,.45,.55,.65,.75,.85,.95):
            admitted=[ok for c,_,ok,_,_ in vals if c>=t]
            curve.append({'confidence_min':t,'coverage':len(admitted)/len(vals),
                          'risk':None if not admitted else 1-mean(admitted)})
        result[arm]['risk_coverage']=curve
        result[arm]['brier_proposal']=mean((c-int(ok))**2 for c,_,ok,_,_ in vals)
        unknown=[c for c,d,_,_,_ in vals if d=='UNKNOWN']
        result[arm]['unknown_fraction']=len(unknown)/len(vals)
    # All adaptive inequalities must independently pass; no held selection of baseline.
    a=metrics['adaptive'];prec=a['correct_active']/max(1,a['active'])
    coverage=a['decisions']/max(1,a['served'])
    cost_gate=all(v['extra_session_ns_ci95'][1]<=0 for v in contrasts.values())
    qual=bool(model['calibration_met'] and coverage>=.55 and prec>=.80 and
        all(v['ci95'][0]>0 for v in contrasts.values()) and
        a['null_interventions']<=min(metrics[x]['null_interventions'] for x in p.MATCHED if x!='adaptive') and
        result['adaptive']['brier_proposal']<=result['fixed_evsi']['brier_proposal']+.01)
    effects={k:{'rows':len(v),'noop_optimal':sum(x[0]==0 for x in v),'mean_best_uplift':mean(x[0] for x in v),
                'mean_signed_actions':[mean(x[1] for x in v),mean(x[2] for x in v)],
                'max_best_uplift':max(x[0] for x in v),
                'strict_active_improvement_cases':sum(x[0]>0 for x in v),
                'noop_active_tie_cases':sum(x[0]==0 and (x[1]==0 or x[2]==0) for x in v),
                'strict_noop_win_cases':sum(x[1]<0 and x[2]<0 for x in v)} for k,v in oracle_rows.items()}
    return {'scope':'bounded constructed DEV, NOT_QUALIFIED','oracle_effects':effects,
            'offline_all_sources_not_available_to_live_policy':dict(offline_conditional),
            'source_value_agreement_not_independence':dict(agreement),'policy_metrics':result,'contrasts':contrasts,
            'adaptive_active_precision':prec,'adaptive_coverage':coverage,
            'quality_gate':qual,'inference_cost_gate':cost_gate,'gate':bool(qual and cost_gate)}
