"""Re-download the public shards, select on development, then rerun held scoring."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cog-051-060'
UPSTREAM = ('https://raw.githubusercontent.com/budzianowski/multiwoz/'
            'fe0c8e65cfcd8462bd33c86e35f21addc84ca82b/data/MultiWOZ_2.2/')

def load(p):
    return json.loads(p.read_bytes())

def main():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for name, digest in load(BASE/'results/dialogue-transfer.json')['source'].items():
            if name.split('/')[0] not in {'test','dev'}:
                continue
            data = urlopen(UPSTREAM+name,timeout=90).read()
            assert hashlib.sha256(data).hexdigest()==digest
            dest = root/name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(data)
        env = dict(os.environ,MULTIWOZ_DIR=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        for mode in ('select','score'):
            subprocess.run([sys.executable,'study.py',mode,'--out',str(root),'--lock',str(root/'selection.json')],
                           cwd=HERE,env=env,check=True)
            if mode=='select':
                assert load(root/'selection.json')['selected']==load(HERE/'selection.json')['selected']
        assert load(root/'results.json')['comparison']==load(HERE/'results.json')['comparison']
        baseline = load(HERE/'traces.json')
        for arm,rows in load(root/'traces.json').items():
            assert [r[:7] for r in rows]==[r[:7] for r in baseline[arm]]
    print('Reproduced selected policies and all five per-turn decision streams.')

if __name__=='__main__':
    main()
