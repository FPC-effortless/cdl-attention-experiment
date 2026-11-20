"""Rebuild permission conditions and score semantic, causal and economic outcomes."""
import json
from collections import Counter
from pathlib import Path
from statistics import median
import numpy as np
from sklearn.metrics import average_precision_score
from boundary import OPERATIONS,HIGH
from language import cases
from transfer import SEEDS

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
def read(p):return json.loads(p.read_bytes())
def summarize(panel,runs):
    result={};arms=sorted({x['arm'] for x in runs})
    by={(v['seed'],v['arm']):v for v in runs}
    for arm in arms:
        sessions=[v for v in runs if v['arm']==arm];allrows=[]
        for s in sessions:
            truth=cases(s['seed'],s['bank'],len(s['rows']))
            for i,(c,r) in enumerate(zip(truth,s['rows'])):
                assert r['gold']==c.a*4+c.t and len(set(r['top5']))==5
                gate=[lambda x,y:x,lambda x,y:y,lambda x,y:x or y,lambda x,y:x and y]
                rule=(0,1+s['seed']%3,1+(s['seed']+1)%3)[i//64]
                if r['status']=='committed':
                    assert c.permitted and gate[rule](*c.flags) and (OPERATIONS[r['proposal']//4] not in HIGH or c.explicit)
                    assert r['authorized'] and not r['bad_permission']
                assert r['semantic_error']==(r['status']=='committed' and r['proposal']!=r['gold'])
                allrows.append(r)
        scoped=[r for r in allrows if r['kind'] in ('plain','echo','conflict')]
        rec=dict(turns=len(allrows),families_n=len(scoped),first1=sum(r['gold']==r['top5'][0] for r in scoped),
                 first5=sum(r['gold'] in r['top5'] for r in scoped),provisional=sum(r['gold']==r['provisional'] for r in allrows),
                 proposal_correct=sum(r['gold']==r['proposal'] for r in allrows),
                 commits=sum(r['status']=='committed' for r in allrows),wrong=sum(r['semantic_error'] for r in allrows),
                 eligible=sum(r['requested_eligible'] for r in allrows),asks=sum(bool(r['asked']) for r in allrows),
                 rolled_nodes=sum(r['rolled'] for r in allrows),work=sum(sum(s['units'].values()) for s in sessions),
                 wall_ms=sum(s['elapsed_ns'] for s in sessions)/1e6,external_teacher_s=sum(s.get('teacher_charge_ns',0) for s in sessions)/1e9)
        base=[by[s['seed'],'primitive'] for s in sessions]
        rec['matched_exact']=all([r['proposal'] for r in s['rows']]==[r['proposal'] for r in b['rows']] and
                                 [r['status'] for r in s['rows']]==[r['status'] for r in b['rows']] for s,b in zip(sessions,base))
        ratios=[b['elapsed_ns']/s['elapsed_ns'] for s,b in zip(sessions,base)]
        rec['speed_by_seed']=ratios
        delta=np.array([sum(z['semantic_error'] for z in s['rows'])-sum(z['semantic_error'] for z in b['rows']) for s,b in zip(sessions,base)])
        rng=np.random.default_rng(80080)
        rec['wrong_delta_mean_ci']=np.quantile(rng.choice(delta,(5000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
        result[arm]=rec
    return result

def main():
    summary={}
    for panel in ('primary','rep','transfer'):
        runs=read(R/panel/'sessions.json') if panel!='transfer' else read(R/'transfer.json')
        summary[panel]=summarize(panel,runs)
    trials=[];econ={};shadow=[]
    for name in ('primary','rep'):
        parts=read(R/name/'subsystems.json')
        assert parts['boundary']['unauthorized_writes']==0
        trials.extend(r for group in parts['causal'].values() for r in group)
        shadow.append(parts['shadow'])
        for s,v in parts['plans'].items():
            for k,x in v.items():
                hot,arm=k.split('/')
                assert x['costs']==v[hot+'/primitive']['costs']
                econ.setdefault(k,[]).append([x['work'],x['elapsed_ns'],v[hot+'/primitive']['elapsed_ns']/x['elapsed_ns']])
    summary['causal']={}
    for silent in (True,False):
        for svc in ('available','missing'):
            for mode in ('active','passive','frozen'):
                rs=[r for r in trials if (r['silent'],r['service'],r['mode'])==(silent,svc,mode)]
                delays=[r['delay'] for r in rs if r['delay'] is not None]
                summary['causal'][f'{silent}/{svc}/{mode}']=dict(n=len(rs),recovered=len(delays),
                    mean_delay=np.mean(delays) if delays else None,errors=sum(r['errors'] for r in rs),
                    cost=sum(r['cost'] for r in rs),probes=sum(r['probes'] for r in rs))
    summary['planning']={k:dict(work=sum(a for a,b,c in v),median_speed=median(c for a,b,c in v)) for k,v in econ.items()}
    summary['shadow']=shadow[0];assert shadow[0]==shadow[1]
    summary['banking']={}
    for k,v in read(R/'authentic-banking.json')['arms'].items():
        rows=v['rows'];unique=[r for r in rows if not r[-1]]
        summary['banking'][k]=dict(n=len(rows),top1=sum(y==p[0] for _,y,p,a,o in rows),top5=sum(y in p for _,y,p,a,o in rows),
            no_overlap_n=len(unique),no_overlap_top5=sum(y in p for _,y,p,a,o in unique),
            admissions=sum(a for _,y,p,a,o in rows),tp=sum(a and y==p[0] for _,y,p,a,o in rows))
    summary['issues']={}
    for arm,v in read(R/'authentic-issues.json')['arms'].items():
        for cohort in ('test','external'):
            rows=[r for r in v['rows'] if r[2]==cohort]
            summary['issues'][arm+'/'+cohort]=dict(n=len(rows),pos=sum(r[3] for r in rows),tp=sum(r[3] and r[5] for r in rows),
                 admitted=sum(r[5] for r in rows),ap=average_precision_score([r[3] for r in rows],[r[4] for r in rows]))
    d=read(R/'authentic-dialogue.json')
    for rows in d.values():assert [r[4] for r in rows]==[r[4] for r in d['verified']]
    summary['dialogue']={k:dict(joint=sum(r[2] for r in v),stale=sum(not r[2] and r[3] for r in v)) for k,v in d.items()}
    acquired=read(R/'admission.json')
    raw_sessions=[dict(s,arm='primitive') for s in acquired['sessions'] if s['arm']=='raw']
    summary['attention_free_admission']=summarize('admission',acquired['sessions']+raw_sessions)
    summary['diagnostic_request']={}
    episodes=[r for group in acquired['causal'].values() for r in group]
    for confounded in (True,False):
        for available in (True,False):
            for mode in ('demand','regular','passive','frozen'):
                rs=[r for r in episodes if (r['confounded'],r['available'],r['mode'])==(confounded,available,mode)]
                summary['diagnostic_request'][f'{confounded}/{available}/{mode}']={k:sum(r[k] for r in rs) for k in ('right','unknown_diagnosis','cost','probes','wrong')}
    correction=[r for group in read(R/'corrections.json').values() for r in group]
    summary['corrections']={}
    for cond in ('legitimate','mirror','late','mistaken','silence'):
        for mode in ('claimed','signed'):
            rs=[r for r in correction if (r[0],r[1])==(cond,mode)]
            assert all((r[4]=='committed')==(cond!='silence') for r in rs)
            summary['corrections'][cond+'/'+mode]=dict(n=len(rs),correct_commits=sum(r[4]=='committed' and r[2]==r[3] for r in rs))
    (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
