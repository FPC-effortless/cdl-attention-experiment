"""Independent October holdout for the state versus timestamp repair."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
ROOT=Path(__file__).resolve().parent
F=Path('/workspace/issue-history-followup')
sys.path.insert(0,str(F))
import acquire,prepare
data=Path('/workspace/cog-next-private/october')
data.mkdir(exist_ok=True)
acquire.DATA=data
acquire.DESIGN=dict(acquire.DESIGN,created=['2026-10-01','2026-10-31'],issue_limit_per_source=18)
acquire.main()
rows=[]
for c in json.loads((data/'cohorts.json').read_text()):
    for i in c['issues']:
        record=json.loads((data/(c['repo'].replace('/','--')+'--'+str(i['number'])+'.json')).read_text())
        actions=[(e['created_at'],e['event']) for e in record['timeline'] if e['event'] in ('reopened','closed')]
        state='closed' if actions and actions[-1][1]=='closed' else 'open'
        search,direct=i,record['direct']
        conflict=state!=search['state']
        date_matches=state=='open' or (actions and search['closed_at']==actions[-1][0])
        # Demand endpoint agreement for a current read and quarantine a conflicting transition time independently.
        current=state if not conflict else None
        terminal=search['closed_at'] if not conflict and date_matches else None
        rows.append(dict(issue=f"{c['repo']}#{i['number']}",timeline=state,search=search['state'],direct=direct['state'],
                         current=current,date_agrees=bool(date_matches),timestamp=terminal,
                         direct_closed_at=direct['closed_at'],updated_search=search['updated_at'],updated_direct=direct['updated_at']))
(ROOT/'results/october-snapshot.json').write_text(json.dumps(rows))
