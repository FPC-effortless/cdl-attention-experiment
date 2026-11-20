"""Re-fit both authentic public benchmarks on a separate CPU from pinned upstream bytes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_bytes())


def acquire(root, url, pins):
    for name, digest in pins.items():
        data = urlopen(url + '/' + name, timeout=90).read()
        assert hashlib.sha256(data).hexdigest() == digest, name
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)


def main():
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        out = work / 'results'
        out.mkdir()
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MULTIWOZ_DIR=str(work/'multiwoz'), EPIC_DATA_DIR=str(work/'epic'))
        expected = load(BASE/'results/kitchen.json')
        acquire(work/'epic', 'https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/'
                'ea8b40457a400c3fffa1c7f406ef3dc169cc2522', expected['source_hashes'])
        dialogue = load(BASE/'results/dialogue-transfer.json')
        acquire(work/'multiwoz', 'https://raw.githubusercontent.com/budzianowski/multiwoz/'
                'fe0c8e65cfcd8462bd33c86e35f21addc84ca82b/data/MultiWOZ_2.2', dialogue['source'])
        for script in ('kitchen.py', 'dialogue_transfer.py'):
            shutil.copyfile(BASE/script, work/script)
            subprocess.run([sys.executable, script], cwd=work, env=env, check=True)
        kitchen = load(out/'kitchen.json')
        for key, arm in expected['arms'].items():
            observed = kitchen['arms'][key]
            for field in ('all', 'new_pair', 'unseen_people', 'masked'):
                assert observed[field] == arm[field], (key, field)
            assert [r[:3] for r in observed['rows']] == [r[:3] for r in arm['rows']], key
        actual = load(out/'dialogue-transfer.json')
        for key, arm in dialogue['slots'].items():
            for method, metrics in arm.items():
                for field in ('tp', 'admitted', 'positives'):
                    assert actual['slots'][key][method][field] == metrics[field], (key, method, field)
        print('Reproduced both kitchen action tables, per-event labels and all eight dialogue operating points.')


if __name__ == '__main__':
    main()
