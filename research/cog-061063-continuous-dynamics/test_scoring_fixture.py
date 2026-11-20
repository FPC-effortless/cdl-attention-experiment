"""Prospective synthetic scorer/adversarial fixtures. No campaign fit, DEV or checkpoint reads."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import score_veritas as sc
BASE=Path(__file__).resolve().parent
EXPECTED='0f775c17040e52d2ea712f1ae42d4ae448899d372535298a6eaddd7513a10b04'
# Only the transport callback is stubbed. Actual scorer consumes synthetic JSONs.
sc.write=lambda x: None
def emit(path,x):
 Path(path).write_text(json.dumps(x))
def main():
 assert hashlib.sha256((BASE/'PREFIT_SHA256.json').read_bytes()).hexdigest()==EXPECTED
 manifest=json.loads((BASE/'PREFIT_SHA256.json').read_text())
 assert hashlib.sha256((BASE/'score_veritas.py').read_bytes()).hexdigest()==manifest['score_veritas.py']
 cwd=Path.cwd()
 with tempfile.TemporaryDirectory() as tmp:
  try:
   os.chdir(tmp);os.environ['TRAINLAB_OUTPUT_DIR']=tmp
   rng=np.random.default_rng(777)
   keys=[str(i) for i in range(64)]
   plans=np.zeros((4,16,2));plans[:,:4]=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])[:,None]*.65
   a=np.zeros((4,16,16));a[3]=2
   perfect=a.tolist(); bad=np.zeros_like(a).tolist()
   g={};t={};rr=[];held={a:{} for a in ('predictive-adaptive','pca-adaptive','narx-adaptive')}
   for i,k in enumerate(keys):
    o=rng.normal(size=(129,16));u=np.zeros((128,2)); rr.append({'id':k,'features':{'o':o.tolist(),'u':u.tolist()}});t[k]=o[::4,:4].tolist()
    z=rng.normal(size=4).tolist()
    for m in held: held[m][k]=z
    g[k]={'shifted':i<32,'checks':{str(s):{'z':z,'branches':[{'u':pl.tolist(),'y':y} for pl,y in zip(plans,perfect)]} for s in (0,4,16)}}
   w=np.zeros((48,4));w[-16:-12]=np.eye(4)
   for m in ('predictive','pca','narx'):np.savez(m+'.npz',w=w,mean=np.zeros(16),std=np.ones(16))
   emit('train.json',{'rows':rr});emit('held-encodings.json',held)
   truth={'evaluation':g,'calibration':t};emit('truth.json',truth)
   preds={m:{k:{str(s): (bad if m=='predictive-adaptive' else perfect) for s in (0,4,16)} for k in keys} for m in sc.ARMS}
   emit('predictions.json',preds);sc.main()
   r=json.loads(Path('scores.json').read_text());p=r['cells']['1/16/8']
   assert p['predictive-adaptive']=={'factual':1.0,'effect':2.0,'interaction':2.0}
   assert all(v==0 for v in p['pca-adaptive'].values())
   assert all(c['ci']==[2.0,2.0] for c in r['effect_comparisons'].values())
   assert not r['063_gate']
   n=0
   def reject():
    nonlocal n
    try:sc.main()
    except AssertionError:n+=1;return
    raise AssertionError('malformed fixture accepted')
   a0=preds[sc.ARMS[0]].pop('0');emit('predictions.json',preds);reject();preds[sc.ARMS[0]]['0']=a0
   preds[sc.ARMS[0]]['extra']=a0;emit('predictions.json',preds);reject();del preds[sc.ARMS[0]]['extra']
   bad[0][0][0]=float('nan');emit('predictions.json',preds);reject();bad[0][0][0]=0.
   emit('predictions.json',preds)
   g['0']['checks']['0']['branches'][0]['u'][0][0]=42;emit('truth.json',truth);reject()
   assert n==4
  finally:os.chdir(cwd)
 print('PASS: perfect zero paired contrasts, known 2.0 main/factorial errors, 95% paired deltas, 4 malformed-input rejections; synthetic only')
if __name__=='__main__':main()
