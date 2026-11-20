import hashlib
import json
import os
import sys
from pathlib import Path

from grounding import run as ground
from panel import run as discover
from system_tests import bridge, persistent, scaling

root=Path(os.environ.get('COG_PRIOR_ROOT','/workspace/delta-authority-048'))
manifest=json.loads(Path('prior-inputs.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
sys.path.insert(0,str(root/'recipe'))
import bridge_scope
from scope import Scope

seed=int(os.environ['TRAINLAB_SEED'])
out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
arm=sys.argv[1]
panel=discover(seed)
language=ground(seed,out,bridge_scope.reader)
routes=bridge(seed,bridge_scope,Scope,root)
selected=routes['results']['integrated' if arm=='candidate' else 'rotate_lrc']
session=persistent(seed,language,selected,bridge_scope.World)
result=dict(seed=seed,arm=arm,panel=panel,language=language,lrc=routes['lrc'],
            route_counts={k:sum(r['exact'] for r in v) for k,v in routes['results'].items()},
            session=session,scaling=scaling(seed))
out.joinpath('result.json').write_text(json.dumps(result,default=lambda v:v.item()))
stable=[language['accuracy_counts'],panel['discovery'],session,routes['lrc']]
out.joinpath('stable.sha256').write_text(hashlib.sha256(json.dumps(stable,sort_keys=True,default=lambda v:v.item()).encode()).hexdigest())
out.joinpath('metrics.json').write_text(json.dumps({'success':sum(t['exact'] for t in session)/len(session)}))
