"""Weight-free paired comparison of native decoding and exact diagnostic probe."""
from hashlib import sha256
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from judge_info_trace import panel,interval,COLORS
S='f868706de68fd723e0e87388969468763cd9274d5fae4739d173cb6107f262df'
PIN='2a364701848636d70e40719b9f85ab1a8bba7f93666954f9280d69f212b06828'
BASE='797243b7e0cf8c118ccbdb4a68b615f15290e2f36e09c96574a6cb1680651c8d'
def need(x,m):
    if not x:raise ValueError(m)
def read(p,pin=None):
    b=Path(p).read_bytes()
    need(pin is None or sha256(b).hexdigest()==pin,'pinned bytes changed '+str(p))
    return json.loads(b)
def measure(z,y,f=None):
    true=z[np.arange(len(y)),y]
    if f is None:
        masked=z.copy();masked[np.arange(len(y)),y]=-np.inf
        correct=true>masked.max(1)
        ce=logsumexp(z,axis=1)-true
    else:
        foil=z[np.arange(len(y)),f]
        correct=true>foil
        ce=np.logaddexp(0,foil-true)
    return correct,ce
def main(native_seal,probe_file,probe_pin):
    n=read('READOUT_SEAL.json',native_seal);p=read(probe_file,probe_pin)
    prior=read('INFO001_PROBE_RESULTS.json',BASE)['stagewise']
    need(n['ledger_sha256']==PIN and n['original_trace_seal_sha256']==S,'readout seal')
    tgt,ids,fams,groups=panel();held=ids['probe_held']
    y=tgt['value'][held];f=tgt['foil'][held];o=np.asarray(held);first=tgt['role'][held]==1
    ix={v:i for i,v in enumerate(held)};gp=[[ix[r] for r in rr] for rr in groups.values()]
    out={}
    for k,r in n['runs'].items():
        seed=int(k.rsplit('-',1)[1]);trace=f'trace-{seed}'
        a=Path('readout_runs')/r['id']; rr=read(a/'run.json',r['receipt_sha256'])
        z=read(a/'output/predictions.json',r['prediction_sha256'])
        need(rr['artifacts']['predictions.json']==r['prediction_sha256'] and
          z['pre_output_sha256']==r['pre_output_sha256'] and z['seed']==seed,'native sealed identity')
        v={w:i+262 for i,w in enumerate(z['tokenizer_words'])}
        zs=np.asarray(z['logits'],dtype=np.float64)[o,:]
        need(zs.shape==(512,287) and np.isfinite(zs).all(),'native logit shape')
        full,fullce=measure(zs,np.array([v[COLORS[j]] for j in y]))
        e=zs[:,[v[x] for x in COLORS]]
        need(p['schema']=='cog-info-001.value-probe-held.v1' and p['class_order']==list(COLORS),'probe classes')
        pr=p['seeds'][trace]
        need(pr['held_row_ids']==held and pr['checkpoint_sha256']==z['checkpoint_sha256'],'probe row lineage')
        pz=np.asarray(pr['log_probs'],dtype=np.float64)
        need(pz.shape==(512,8) and np.isfinite(pz).all() and np.max(np.abs(logsumexp(pz,axis=1)))<1e-10,'probe normalization')
        s={}
        for name,foils in (('eight',None),('two',f)):
            nh,nce=measure(e,y,foils);ph,pce=measure(pz,y,foils)
            check=prior[trace]['value']['q_plus_pre_output_hidden']['metrics']
            if name=='eight':
                need(int(ph.sum())==check['correct'] and abs(float(pce.mean())-check['ce_nats'])<1e-10,'probe metric mismatch')
            else:
                need(int(ph.sum())==check['relative_correct'] and abs(float(pce.mean())-check['relative_ce_nats'])<1e-10,'conditional mismatch')
            resc=ph&~nh; dam=nh&~ph
            s[name]=dict(native_correct=int(nh.sum()),probe_correct=int(ph.sum()),rescues=int(resc.sum()),damages=int(dam.sum()),
              first_rescues=int((resc&first).sum()),first_damages=int((dam&first).sum()),
              second_rescues=int((resc&~first).sum()),second_damages=int((dam&~first).sum()),
              native_ce_nats=float(nce.mean()),probe_ce_nats=float(pce.mean()),
              probe_minus_native_ce_grouped_95_nats=interval([float((pce[i]-nce[i]).mean()) for i in gp]))
        out[k]=dict(native_strict_287_correct=int(full.sum()),native_full_ce_nats=float(fullce.mean()),matching=s,rows=512,
             checkpoint_sha256=z['checkpoint_sha256'],pre_output_parity_sha256=z['pre_output_sha256'])
    pth=Path('NATIVE_READOUT_COMPARISON.json')
    with pth.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
    print(sha256(pth.read_bytes()).hexdigest())
if __name__=='__main__':main(*sys.argv[1:])
