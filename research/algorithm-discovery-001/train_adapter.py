"""Run the existing, pinned REPAIR-009 full-budget trainer on the new candidates.

Requires the original repository checkout at the named commit and its prepared
WikiText TRAIN/dev data. Does not fetch official TEST. No job is launched on import.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from contextual_cog import make,MODES

ARMS=('HH','A0',*MODES)
SEEDS=(9001,9011,9029,9041)
PARENT_BLOBS={'model.py':'31c3555521469425d884c700aed01b48c4fab50e','train009.py':'5d0d726338a3fd635c25a9fbbd7e7a1f4cec822d','data009.py':'08afe12ab312f04b23b4b841fb294d2a9895c560'}
HERE=Path(__file__).resolve().parent

def blob(p):
    b=p.read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def terminal_gate(trainer,root):
    # Check every terminal weight receipt AND source/data/order lock before
    # allowing access to official validation. This is a mechanical custody gate,
    # not a claim of independent holdout enforcement or scientific qualification.
    expected_source=trainer.source_lock();data_hashes=set();orders={}
    for arm in ARMS:
        for seed in SEEDS:
            folder=root/arm/str(seed)/'stage4';p=folder/'model.safetensors';r=json.loads((folder/'model_receipt.json').read_text());lock=r['lock']
            assert r['step']==trainer.STEPS and r['targets']==trainer.STEPS*trainer.B*trainer.T
            assert lock['arm']==arm and lock['seed']==seed and lock['source']==expected_source
            assert sha(p)==r['sha256'];data_hashes.add(lock['data']);orders.setdefault(seed,set()).add(lock['data_order'])
    assert len(data_hashes)==1 and all(len(s)==1 for s in orders.values()),'data/order mismatch'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','validate','gate']);p.add_argument('--parent',type=Path,required=True,help='Path to pinned research/math-repair-009 directory');p.add_argument('--data',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--stage',type=int,choices=range(1,5));p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    parent=a.parent.resolve();a.out_root=a.out_root.resolve();a.data=str(a.data.resolve())
    for name,expected in PARENT_BLOBS.items():assert blob(parent/name)==expected,(name,'wrong source revision')
    assert blob(HERE/'model.py')==PARENT_BLOBS['model.py']
    sys.path.insert(1,str(parent));import train009 as trainer
    # The parent order generator accepts any integer seed; only the trainer's
    # explicit membership assertion requires the fourth predeclared seed.
    trainer.SEEDS=SEEDS;trainer.ARMS=ARMS;trainer.make=make
    # Parent evaluate can return NumPy scalars; make serialized curve metadata
    # safe for the parent's weights_only=True continuation. No numeric change.
    original_eval=trainer.evaluate
    def evaluate(*args,**kwargs):
        e=original_eval(*args,**kwargs);e['bpb']=float(e['bpb']);e['accuracy']=float(e['accuracy']);e['targets']=int(e['targets']);return e
    trainer.evaluate=evaluate
    original_lock=trainer.source_lock
    def source_lock():
        d=original_lock();d.update({f'algorithm001/{name}':sha(HERE/name) for name in ('model.py','contextual_cog.py','train_adapter.py','PROTOCOL.md')});return d
    trainer.source_lock=source_lock
    root=a.out_root
    if a.mode=='gate':
        assert a.arm is None and a.seed is None and a.stage is None
        terminal_gate(trainer,root)
        import data009
        data009.verify(a.data)
        for arm in ARMS:
            for seed in SEEDS:
                r=json.loads((root/arm/str(seed)/'stage4'/'model_receipt.json').read_text());lock=r['lock']
                assert lock['data']==sha(Path(a.data)/'manifest.json')
                assert lock['data_order']==hashlib.sha256(data009.order(a.data,seed).tobytes()).hexdigest()
        print('TERMINAL_GATE_PASS 24',flush=True);return
    assert a.arm is not None and a.seed is not None
    armroot=root/a.arm/str(a.seed)
    if a.mode=='train':
        import recover009
        recover009.corpus(a.data)
        assert a.stage is not None;folder=armroot/f'stage{a.stage}';assert not folder.exists(),'refusing to overwrite existing stage evidence';a.out=str(folder);a.resume=str(armroot/f'stage{a.stage-1}'/'checkpoint.pt') if a.stage>1 else None;trainer.train(a)
    else:
        assert a.stage is None;terminal_gate(trainer,root);a.resume=str(armroot/'stage4'/'model.safetensors');a.out=str(armroot/'validation');assert not Path(a.out).exists(),'refusing to overwrite existing held evaluation';trainer.final(a)
if __name__=='__main__':main()
