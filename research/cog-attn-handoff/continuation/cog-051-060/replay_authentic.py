import json
from pathlib import Path
import shutil
from trainlab.runner import run_experiment,verify_run
HERE=Path(__file__).resolve().parent
P=Path('/workspace/cog-next-private')
source=P/'authentic'; target=P/'authentic-replay';target.mkdir(exist_ok=True)
receipts=[]
for seed in (67003,67009,67021):
    cfg=json.loads((source/f'{seed}.json').read_text())
    for f in cfg['inputs']:shutil.copyfile(source/f,target/f)
    path=target/f'{seed}.json';path.write_text(json.dumps(cfg))
    run=run_experiment(path,target/'runs')
    assert run['status']=='succeeded' and verify_run(target/'runs'/run['id'])['valid']
    primary=json.loads((HERE/'results'/str(seed)/'authentic-receipt.json').read_text())
    assert run['inputs']==primary['inputs']
    for f in ('issue-results.json','issue-predictions.json'):
        assert run['artifacts'][f]==primary['artifacts'][f]
    receipts.append({'seed':seed,'primary':primary['id'],'replay':run['id'],'matched_predictions':run['artifacts']['issue-predictions.json']})
(HERE/'results/authentic-replays.json').write_text(json.dumps(receipts,indent=2))
