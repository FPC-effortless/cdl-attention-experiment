"""Stage predictions using an archived validation cutoff and pay for delayed observation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import beta

HERE=Path(__file__).resolve().parent
F=Path('/workspace/issue-history-followup')
sys.path.insert(0,str(F))
from github_source import API,fetch

def call(issue):
    repo,num=issue.split('#')
    start=perf_counter()
    page=fetch(f'{API}/repos/{repo}/issues/{num}/timeline',{'per_page':100})
    raw=page.json(); pages=1
    while 'next' in page.links:
        page=fetch(page.links['next']['url']);raw.extend(page.json());pages+=1
    rows=[]
    for e in raw:
        kind=e.get('event')
        if kind=='commented':
            who=e.get('user') or {}
            if who.get('type')!='User' or not who.get('id') or (who.get('login') or '').endswith('[bot]'): continue
        elif kind in ('closed','reopened'): who=e.get('actor') or {}
        else: continue
        rows.append((kind,e['id'],who.get('id'),e['created_at'],e.get('updated_at')))
    found={}
    for i,(kind,key,who,when,edited) in enumerate(rows):
        if kind!='commented' or when!=edited: continue
        after=rows[i+1] if i+1<len(rows) else None
        delta=(datetime.fromisoformat(after[3])-datetime.fromisoformat(when)).total_seconds() if after else None
        found[key]={'action':after[0] if after and after[0]!='commented' and after[2]==who and 0<delta<=900 else None,
                    'await_seconds':min(900,max(0,delta)) if delta is not None else 900}
    return issue,(found,pages,perf_counter()-start)

def main():
    p=json.loads((HERE/'results/67003/issue-predictions.json').read_text())
    head=[r for r in p if r[0]=='pooled_flat']
    pos=sorted((r[5] for r in head if r[1]=='validation' and r[4]),reverse=True)
    t=pos[int(np.ceil(.8*len(pos)))-1]
    staged=[r for r in head if r[1]!='validation' and r[5]>=t]
    with ThreadPoolExecutor(8) as pool: live=dict(pool.map(call,sorted({r[2] for r in staged})))
    data=json.loads((F/'recipe/dataset.json').read_text())
    legal={r['comment']:('reopened' if r['closed'] else 'closed') for r in data}
    summary={}; evidence=[]
    for split in ('test','external'):
        subset=[r for r in staged if r[1]==split]
        countries={r[2] for r in subset}
        tp=fp=absent=0; admitted={};wait=[]
        for _,_,issue,key,y,score,_ in subset:
            answer=live[issue][0].get(key)
            if not answer: absent+=1;continue
            act=answer['action']==legal[key]
            tp+=bool(act and y);fp+=bool(act and not y);wait.append(answer['await_seconds'])
            if act: admitted[issue]=admitted.get(issue,True) and bool(y)
            evidence.append([split,issue,key,y,score,act,answer['await_seconds']])
        n=sum(r[4] for r in head if r[1]==split)
        good=sum(admitted.values())
        summary[split]=dict(positives=n,staged=len(subset),tp=tp,fp=fp,unknown=absent,recall=tp/n,
            precision=tp/(tp+fp) if tp+fp else None,issue_lower=float(beta.ppf(.05,good,len(admitted)-good+1)) if good else 0,
            distinct_issues=len(countries),page_requests=sum(live[k][1] for k in countries),
            aggregate_request_seconds=sum(live[k][2] for k in countries),median_observation_delay_seconds=float(np.median(wait)))
    (HERE/'results/confirmation.json').write_text(json.dumps(dict(validation_threshold=t,summary=summary,rows=evidence)))
if __name__=='__main__': main()
