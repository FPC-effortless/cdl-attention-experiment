"""Fixed-budget structural search using exact parent trainer and corpus."""
import argparse,hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'math-repair-009'
from variants import ARMS,fresh
SEEDS=(9301,9311,9329)
EXPECTED={'model.py':'31c3555521469425d884c700aed01b48c4fab50e','train009.py':'5d0d726338a3fd635c25a9fbbd7e7a1f4cec822d','data009.py':'08afe12ab312f04b23b4b841fb294d2a9895c560'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blob(p):
    b=p.read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    assert all(blob(PARENT/n)==h for n,h in EXPECTED.items());assert blob(HERE/'model.py')==EXPECTED['model.py'];assert not a.out.exists(),'refusing to overwrite evidence'
    sys.path.insert(1,str(PARENT));import train009 as trainer;import data009,recover009
    recover009.corpus(a.data);assert sha(a.data/'manifest.json')=='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'
    assert trainer.T==768 and trainer.B==16 and trainer.STEPS==8192 and trainer.STAGE==2048
    trainer.STEPS=trainer.STAGE=1024;trainer.SEEDS=SEEDS;trainer.ARMS=ARMS;trainer.make=fresh;trainer.order=lambda root,seed:data009.order(root,seed)[:1024]
    trainer.lr=lambda i:.003*(i+1)/20 if i<20 else .003*(.1+.45*(1+math.cos(math.pi*(i-20)/(1023-20))))
    original_eval=trainer.evaluate
    def evaluate(*args,**kwargs):
        d=original_eval(*args,**kwargs);d['bpb']=float(d['bpb']);d['accuracy']=float(d['accuracy']);d['targets']=int(d['targets']);return d
    trainer.evaluate=evaluate;original_lock=trainer.source_lock
    def source_lock():
        d=original_lock();d.update({'hh_u_search_001/'+n:sha(HERE/n) for n in ('model.py','variants.py','screen.py','preflight.py','train_search.py','aggregate.py','PROTOCOL.md','TRAIN_PROTOCOL.md')});return d
    trainer.source_lock=source_lock;a.stage=1;a.resume=None;a.data=str(a.data);a.out=str(a.out);trainer.train(a)
    out=Path(a.out);r=json.loads((out/'model_receipt.json').read_text());c=json.loads((out/'receipt.json').read_text());assert r['step']==c['step']==1024 and r['targets']==c['targets']==12582912 and c['terminal_stage'];assert r['lock']==c['lock'] and r['lock']['source']==source_lock()
    import torch
    from safetensors.torch import load_file
    assert sha(out/'checkpoint.pt')==c['sha256'] and sha(out/'model.safetensors')==r['sha256'];ck=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu');w=load_file(str(out/'model.safetensors'));assert all(torch.equal(v,w[k]) for k,v in ck['weights'].items()) and ck['optimizer']['state'] and torch.is_tensor(ck['rng']);assert ck['lock']==r['lock'] and ck['step']==1024
    d=json.loads((out/'development.json').read_text());assert d['targets']==5029184 and d['bpb']==r['dev_bpb']
    (out/'result.json').write_text(json.dumps(dict(arm=a.arm,seed=a.seed,bpb=d['bpb'],eval_targets=d['targets'],parameters=r['parameters'],targets=r['targets'],step=r['step'],training_seconds=r['training_seconds'],checkpoint_sha256=c['sha256'],model_sha256=r['sha256'],lock=r['lock'],terminal=True,reload_verified=True),indent=2));print((out/'result.json').read_text(),flush=True)
if __name__=='__main__':main()
