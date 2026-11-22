"""Model-free sealed feature-rank judging. No task outcomes or training."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from hashlib import sha256
from pathlib import Path
import json,sys
import numpy as np
from trainlab.runner import verify_run

P=Path('rank_admissions');R=Path('rank_runs')
PIN='d6d1b4fd5328f408659786699c45de860bf41b329ba82c9a3d76e1c6a5ddd729'
def h(p):return sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p,pin=None):
    b=p.read_bytes();need(pin is None or sha256(b).hexdigest()==pin,'digest changed')
    return json.loads(b)
def save(p,d):
    with p.open('x') as f:json.dump(d,f,allow_nan=False,indent=2,sort_keys=True);f.write('\n')
    print(h(p))
def seal():
    led=read(P/'ledger.json',PIN);slots={}
    need(set(led)=={'rank-1171','rank-1371'},'exact roster')
    for label,c in led.items():
        case=P/label;r=read(case/'cli-output.json');rp=R/r['id']
        need(verify_run(rp)['valid'] and r==read(rp/'run.json') and r['status']=='succeeded' and
             r['inputs']==c['inputs'] and r['seed']==c['seed'] and r['name']==label and 0<r['elapsed_seconds']<=45
             and r['metrics']==dict(model_forwards=664,optimizer_updates=0,feature_rows_per_arm=10240),'native exact run')
        need(h(case/'recipe.json')==c['recipe_sha256'] and read(case/'recipe.json')==read(rp/'experiment.json')
             and all(h(case/k)==v for k,v in c['inputs'].items()),'registry mismatch')
        need(set(r['artifacts'])=={'FEATURES.f32le','geometry.json'},'feature artifacts')
        slots[label]=dict(id=r['id'],receipt_sha256=h(rp/'run.json'),artifacts=r['artifacts'])
    need(len({r['id'] for r in slots.values()})==2,'duplicate')
    save(Path('RANK_SEAL.json'),dict(ledger_sha256=PIN,results=slots))

def ranks(s,pivot=None):
    base=s[0] if pivot is None else pivot
    return {f'{t:g}':int(sum(s>base*t)) for t in (1e-6,1e-4)}
def score(seal_pin):
    led=read(P/'ledger.json',PIN);s=read(Path('RANK_SEAL.json'),seal_pin)
    need(s['ledger_sha256']==PIN and set(s['results'])==set(led),'sealed roster')
    out={}
    for label,p in s['results'].items():
        d=R/p['id'];r=read(d/'run.json',p['receipt_sha256']);a=p['artifacts'];c=led[label]
        need(r['artifacts']==a,'native seals')
        g=read(d/'output/geometry.json',a['geometry.json'])
        need(g['schema']=='cog-role-repair-004.witness.v1' and g['seed']==c['seed'] and
             g['actor_sha256']==c['actor_sha256'] and g['checkpoint_sha256']==c['checkpoint_sha256'] and
             g['arms']==['tap_compare','literal_control'] and g['shape']==[2,2560,4,68] and
             g['features_sha256']==a['FEATURES.f32le'] and g['zero_parity_rows_per_arm']==g['earlier_parity_rows_per_arm']==64,'geometry')
        p=(d/'output/FEATURES.f32le').read_bytes()
        need(sha256(p).hexdigest()==a['FEATURES.f32le'] and len(p)==2*2560*4*68*4,'binary feature identity')
        x=np.frombuffer(p,dtype='<f4').reshape(2,10240,68).astype(np.float64)
        need(np.isfinite(x).all(),'nonfinite feature')
        base=np.concatenate((x[0,:,:32],x[0,:,64:]),axis=1)
        need(np.array_equal(base,np.concatenate((x[1,:,:32],x[1,:,64:]),axis=1)),'common feature mismatch')
        need(np.array_equal(x[0,:,64:],np.tile(np.eye(4),(2560,1))) and
             len(np.unique(x[0,:,:32],axis=0))==16,'literal names or isolated lag changed')
        u,sv,_=np.linalg.svd(base,full_matrices=False)
        b=ranks(sv)
        arm={}
        for name,m in zip(g['arms'],x,strict=True):
            f=np.linalg.svd(m,compute_uv=False)
            rnk=ranks(f)
            residual={}
            for tk,k in b.items():
                z=m[:,32:64]; z=z-u[:,:k]@(u[:,:k].T@z)
                e=np.linalg.svd(z,compute_uv=False)
                residual[tk]=int(sum(e>f[0]*float(tk)))
            arm[name]=dict(full_rank=rnk, source_rank_increment={k:rnk[k]-b[k] for k in b},
                      residual_source_rank=residual, relative_full_singular_values=(f/f[0]).tolist(),
                      nonzero_probe_changed_rows=g['probe_changed_rows'][name])
        out[label]=dict(run_id=r['id'],common_feature_rank=b,common_normalized_singular_values=(sv/sv[0]).tolist(),arms=arm,fit_updates=0)
    save(Path('RANK_RESULTS.json'),out)
if __name__=='__main__':
    if sys.argv[1]=='seal':seal()
    elif sys.argv[1]=='score':score(sys.argv[2])
    else:raise ValueError('unsupported')
