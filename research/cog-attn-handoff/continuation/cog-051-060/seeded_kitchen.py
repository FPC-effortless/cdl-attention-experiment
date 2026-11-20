"""Run the fixed-solver replication alongside the frozen primary kitchen fit."""
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
D = Path(os.environ.get('EPIC_DATA_DIR', '/workspace/cog-next-private/epic'))


def main():
    with TemporaryDirectory() as temp:
        work = Path(temp)
        (work/'results').mkdir()
        for path in D.glob('*.csv'):
            (work/path.name).symlink_to(path.resolve())
        copyfile(ROOT/'kitchen.py', work/'kitchen.py')
        env = dict(os.environ, KITCHEN_SOLVER_SEED='5152', EPIC_DATA_DIR=str(work),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, 'kitchen.py'], cwd=work, env=env, check=True)
        copyfile(work/'results/kitchen.json', ROOT/'results/kitchen-seeded.json')
        dest = ROOT/'checkpoints/kitchen'
        dest.mkdir(parents=True, exist_ok=True)
        copyfile(work/'models.joblib', dest/'seeded.joblib')


if __name__ == '__main__':
    main()
