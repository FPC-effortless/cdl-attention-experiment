"""Process authentic comment arrivals and later confirmations in causal order in a shadow ledger."""
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0,'/workspace/delta-authority-048/recipe')
from state_store import World
R=Path(__file__).resolve().parent/'results'
rows=json.loads(Path('/workspace/issue-history-followup/recipe/dataset.json').read_text())
comments={r['comment']:r for r in rows if r['strict']}
conf=json.loads((R/'confirmation.json').read_text())
proposals={k:(split,issue,confirmed,lag) for split,issue,k,y,p,confirmed,lag in conf['rows']}
stream=[]
for key,r in comments.items():
    stream.append((r['stamp'],1,key))
    if key in proposals:stream.append((r['stamp']+proposals[key][-1],2,key))
pending={};counts=Counter();examples=[]
with TemporaryDirectory() as folder:
    db=World(Path(folder)/'ledger.sqlite')
    for t,phase,key in sorted(stream):
        r=comments[key];counts['comments' if phase==1 else 'verification_arrivals']+=1
        if phase==1:
            if key in proposals:pending[key]=(db.epoch(),t)
        else:
            split,issue,confirmed,latency=proposals[key]
            epoch,start=pending.pop(key);assert start<=t
            if confirmed:
                patch={issue:'open' if r['closed'] else 'closed'}
                assert db.commit(str(key),epoch,patch)=='committed'
                counts['committed']+=1;counts['incorrect_behavioral_updates']+=1-r['target']
                examples.append([key,start,t,patch])
            else:counts['expired']+=1
    assert not pending and db.db.execute('select count(*) from journal').fetchone()[0]==len(examples)
    db.db.close()
(R/'authentic-session.json').write_text(json.dumps(dict(counts=counts,confirmed=examples)))
