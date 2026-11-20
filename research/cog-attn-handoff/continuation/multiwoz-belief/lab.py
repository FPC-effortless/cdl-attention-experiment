"""Register the protocol in Research Lab OS and replay from two TrainLab workspaces."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from research_lab.core import Controller
from research_lab.evidence import put_node, link
from trainlab.runner import run_experiment, verify_run

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cog-051-060'
LAB = Path(os.environ['BELIEF_LAB_DIR'])
FILES = ['study.py','belief.py','PROTOCOL.md','selection.json']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    lab = Controller(LAB/'os')
    lab.program('belief','Reduce stale dialogue reads with isolated verified commits',600,0)
    pin = put_node(lab,'belief.protocol','experiment',{name:digest(HERE/name) for name in FILES})
    os.environ['COG_CYCLE_DIR'] = str(BASE)
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    records = []
    for label in ('primary','replay'):
        folder = LAB/label
        folder.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            shutil.copyfile(HERE/name,folder/name)
        recipe = dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
                      name='isolated-dialogue-belief',description='Calibrated persistent state read',
                      command=[sys.executable,'study.py','score'],seed=61051,timeout_seconds=300,
                      inputs=FILES,artifacts=['results.json','traces.json'],metrics={'joint':'max'},
                      source_kind='AUTHENTIC_CANDIDATE',split_identity='multiwoz22-official-test-2-hotel-slots',
                      budget={'cpu':1})
        path = folder/'recipe.json'
        path.write_text(json.dumps(recipe))
        record = run_experiment(path,folder/'runs')
        directory = folder/'runs'/record['id']
        assert record['status']=='succeeded' and verify_run(directory)['valid']
        records.append({k:record[k] for k in ('id','artifacts','inputs','elapsed_seconds','metrics')})
        if label=='primary':
            for name in recipe['artifacts']:
                shutil.copyfile(directory/'output'/name,HERE/name)
    assert records[0]['artifacts']==records[1]['artifacts']
    receipt = dict(protocol=pin,runs=records,clean_directory_replay=True)
    put_node(lab,'belief.observation','evidence',receipt)
    link(lab,'belief.observation','tests','belief.protocol')
    (HERE/'lab-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':
    main()
