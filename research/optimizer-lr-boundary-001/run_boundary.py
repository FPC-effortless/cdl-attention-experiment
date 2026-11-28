"""Extend only AdamW peak rates; leave original optimizer campaign untouched."""
import argparse,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'optimizer-discovery-001'
EXPECTED={'run.py':'79e047439059ce1779cb60a8da688ecec91a7eb5','optimizers.py':'5595ffa8e441b6d34ba254ea5b0e408f8d40e165','PROTOCOL.md':'99a2b2392337ed4fc5a50b25143d660500d7ae66','aggregate.py':'f8a37ea3b9c12a963df66e69204adc5b187fb7fd'}
def blob(p):
    b=p.read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rate',type=float,choices=(.002,.003),required=True);p.add_argument('--seed',type=int,choices=(9101,9109),required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    assert all(blob(PARENT/n)==h for n,h in EXPECTED.items()),'parent source changed'
    sys.path.insert(0,str(PARENT));import run as core
    assert core.STEPS==1024 and core.T==768 and core.B==16 and core.SEEDS==(9101,9109)
    assert core.RATES['adamw']==(.0003,.0006,.0012)
    core.RATES['adamw']=(.0012,.002,.003)
    original=core.sources
    def locked(recipe):
        d=original(recipe);d.update({'lr_boundary_001/'+n:sha(HERE/n) for n in ('run_boundary.py','peak_check.py','aggregate_boundary.py','PROTOCOL.md')});return d
    core.sources=locked
    sys.argv=[str(PARENT/'run.py'),'--recipe','adamw','--rate-index',str(core.RATES['adamw'].index(a.rate)),'--seed',str(a.seed),'--data',str(a.data),'--out',str(a.out),'--threads',str(a.threads)]
    core.main()
if __name__=='__main__':main()
