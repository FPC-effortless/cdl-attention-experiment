from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
F=Path('/workspace/issue-history-followup')
sys.path.insert(0,str(F))
from github_source import API,fetch
sys.path.insert(0,str(F/'recipe'))
from wr006 import History
source=json.loads((F/'recipe/snapshots.json').read_text())
outputs={r['issue']:r for r in json.loads((HERE/'results/snapshot.json').read_text())}
stats=Counter()
cases=[]
probes=[]
for row in source:
    key=row['issue']
    r=outputs[key]
    if r['disposition']!='consistent' or int(hashlib.sha256(key.encode()).hexdigest(),16)%25==0: probes.append(row)
    anchor=r.get('anchor')
    events=[(t,1,j,v) for j,(t,v,eid) in enumerate(row['actions'])]
    if anchor:
        t=int(datetime.fromisoformat(anchor['validFrom']).timestamp())
        events.append((t,2,len(events),1))
    queries=set(row['comment_times']+[row['created']]+[t for t,*_ in events])
    if anchor: queries.update([t-1,t,t+1])
    coords={t:i for i,t in enumerate(sorted(queries|{e[0] for e in events}))}
    tree=History(len(coords))
    for t,phase,j,v in events: tree.insert(key,coords[t],phase,v)
    for cutoff in (1,2):
        for when in queries:
            available=[(t,phase,j,v) for t,phase,j,v in events if t<=when and phase<=cutoff]
            expected=max(available)[-1] if available else 0
            assert max(0,tree.lookup(key,coords[when],cutoff)[0])==expected
            stats['historical_checks']+=1
    stats['issues']+=1
def cross(row):
    repo,num=row['issue'].split('#')
    page=fetch(f'{API}/repos/{repo}/issues/{num}/events',{'per_page':100})
    events=page.json()
    while 'next' in page.links:
        page=fetch(page.links['next']['url'])
        events.extend(page.json())
    actual=[e['id'] for e in events if e['event'] in ('closed','reopened')]
    return {'issue':row['issue'],'flagged':outputs[row['issue']]['disposition']!='consistent','equal_event_ids':actual==[v[-1] for v in row['actions']]}
with ThreadPoolExecutor(8) as pool: cases=list(pool.map(cross,probes))
(HERE/'results/chronology.json').write_text(json.dumps({'counts':stats,'crosschecks':cases},indent=2))
