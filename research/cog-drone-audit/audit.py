"""Read-only scientific replay of the two developmental archives. No model fitting."""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile, zipfile, os, shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ORIG=ROOT/'original'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(root):
    expected={}
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        h, name=line.split(None,1); expected[name]=h
    actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and '__pycache__' not in p.parts}
    assert actual==expected, (set(expected)-set(actual),set(actual)-set(expected),[k for k in expected if k in actual and expected[k]!=actual[k]])
    return len(actual)
def import_module(path, name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def synthetic(root):
    m=import_module(root/'experiment.py','synthetic_'+root.name)
    r=json.loads((root/'results.json').read_text()); attention=[x['name'] for x in r['fits'] if x['name'].startswith('Window')][0]
    fac={'COG-inspired masked bank':lambda:m.Bank(True),'Dense bank ablation':lambda:m.Bank(False),'GRU parameter-near':lambda:m.GRU(19),'GRU state-matched':lambda:m.GRU(m.N*m.H),attention:lambda:m.Attention(16)}
    # Extended script uses the same Attention constructor? Verify from archived code.
    tests={'nominal':m.data(8101,96,512,.75),'sparse':m.data(8102,96,512,.15),'sparse_fast_drift':m.data(8103,96,512,.15,drift=.055,shared=.025)}
    maximum=0.;prefix=0.; rows=[]
    with torch.no_grad():
        for row in r['fits']:
            name=row['name'];model=fac[name]();model.load_state_dict(torch.load(root/(name.replace(' ','_')+'_'+str(row['seed'])+'.pt'),weights_only=True,map_location='cpu'));model.eval();assert sum(p.numel() for p in model.parameters())==row['params']
            for k,(x,a,y) in tests.items():
                pred=torch.cat([model(xx,aa) for xx,aa in zip(x.split(8),a.split(8))]);got=m.metrics(pred,y);ref=row['tests'][k];d=max(abs(got['rmse']-ref['rmse']),max(abs(a-b) for a,b in zip(got['trajectory_rmse'],ref['trajectory_rmse'])));maximum=max(maximum,d);assert d<2e-5,(name,k,d)
                p=float((model(x[:2,:129],a[:2,:129])-pred[:2,:129]).abs().max());prefix=max(prefix,p);assert p<1e-4,(name,k,p)
            rows.append((name,row['seed']))
        for name,panels in r['classical'].items():
            for k,(x,a,y) in tests.items():
                got=m.metrics(m.classical(x,'KF' if name.startswith('Kalman') else 'last'),y);ref=panels[k];assert abs(got['rmse']-ref['rmse'])<2e-5
    assert len(rows)==15 and len(set(rows))==15
    return dict(fits=len(rows),panels=3*len(rows),max_metric_delta=maximum,max_prefix_delta=prefix)

def recorded(root):
    sys.path.insert(0,str(root));import run as r;import supplementary as s
    D,g,meta=r.load();R=json.loads((root/'results.json').read_text());S=json.loads((root/'supplementary.json').read_text());F={**r.FACT,**s.FACT}; failures=[]; maximum=0.;batch_stream=0;rows=[];replayed_classical=0;stats=[];nchecks=0
    assert len(R['fits'])==54 and len(S)==27 and len(list((root/'checkpoints').glob('*.pt')))==81
    assert len(set(x['blob_sha'] for x in meta))==12 and meta==R['source']
    assert digest(root/'PROTOCOL.md')==R['protocol_sha256']
    for i,fold in enumerate(r.GROUPS):
        z=R['folds'][fold];tr,va,te=[np.array(z[k]) for k in ['train','val','test']];assert len(tr)==6 and len(va)==2 and len(te)==4;assert len(set(tr)|set(va)|set(te))==12;assert all(g[te]==fold) and all(g[tr]!=fold) and all(g[va]!=fold)
        mean=D[tr].mean((0,1));std=D[tr].std((0,1));assert np.array_equal(mean,np.array(z['mean'],np.float32)) and np.array_equal(std,np.array(z['std'],np.float32))
        d=(D-mean)/std;dev=r.panels(d[va],1000+i);test=r.panels(d[te],2000+i)
        baseline=r.tune(d[tr],dev,test,std)
        for name,x in baseline.items():
            for k,sc in x['tests'].items():
                assert abs(sc['nrmse']-R['classical'][fold][name]['tests'][k]['nrmse'])<2e-5
                replayed_classical+=1
        for row in [x for x in R['fits']+S if x['fold']==fold]:
            name=row['name'];seed=row['seed'];model=F[name]();ck=torch.load(root/'checkpoints'/f'{fold}_{seed}_{name.replace(" ","_")}.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['model']);model.eval();assert ck['name']==name and ck['seed']==seed and ck['fold']==fold;assert ck['train_mean']==z['mean'] and ck['train_std']==z['std'];assert ck['calibration_q']==row['calibration_q'];assert sum(p.numel() for p in model.parameters())==row['params']
            with torch.no_grad():
                for k,(x,l,y,m) in test.items():
                    mu,sd=model(x);sc=r.score(mu,sd,y,m,std,row['calibration_q']);ref=row['tests'][k];delta=max(abs(sc[q]-ref[q]) for q in ['nrmse','nll','v_rmse','a_rmse','coverage90','interval_width_v','interval_width_a']);delta=max(delta,max(abs(a-b) for a,b in zip(sc['per_flight_nrmse'],ref['per_flight_nrmse'])));maximum=max(delta,maximum);assert delta<2e-5,(name,k,delta)
                    # Hidden-target independence and prefixes.
                    alt=y.numpy().copy();alt[~m.numpy()]=1234.;xx,*_=r.inputs(alt,m.numpy());assert torch.equal(x,xx)
                    p=model(x[:2,:47]);assert max(float((a-b[:2,:47]).abs().max()) for a,b in zip(p,(mu,sd)))<1e-4
                    if hasattr(model,'step'):
                        st=model.init(len(x));mus=[];sds=[]
                        for j in range(x.shape[1]):
                            old=st;(a,b),st=model.step(x[:,j],st);mus.append(a);sds.append(b)
                            if name in ['COG-HOLD','COG-DUAL']:
                                mask=~x[:,j,2:4].bool();nchecks+=int(mask.sum())*model.h;assert torch.equal(old[0][mask],st[0][mask])
                        err=max(float((torch.stack(mus,1)-mu).abs().max()),float((torch.stack(sds,1)-sd).abs().max()));batch_stream+=1
                        if err>=1e-4:failures.append(dict(fold=fold,seed=seed,name=name,rate=k,max_abs=err))
                    rows.append((fold,seed,name,k))
                    stats.append(dict(fold=fold,seed=seed,name=name,rate=k,rmse=sc['nrmse'],per_flight=sc['per_flight_nrmse']))
    assert len(set(rows))==162
    # Classify rather than silently fix historical behavior. The known supplementary defect must be detected.
    assert failures and {f['name'] for f in failures}=={'Dual gated evolution ablation'},failures
    return dict(fits=81,panels=len(rows),classical_panels=replayed_classical,max_metric_delta=maximum,streaming_panels=batch_stream,streaming_failures=failures,protected_checks=nchecks,protected_violations=0,statistics=stats)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    report={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'manifest':{},'synthetic':{},'recorded':None}
    for name in ['COG-DRONE-001','COG-DRONE-002-006']:report['manifest'][name]=manifest(ORIG/name)
    for name,dir in [('100',ORIG/'COG-DRONE-001'),('500',ORIG/'COG-DRONE-001'/'extended')]:report['synthetic'][name]=synthetic(dir)
    report['recorded']=recorded(ORIG/'COG-DRONE-002-006')
    (ROOT/'audit_results.json').write_text(json.dumps(report,indent=2)+'\n')
    brief={**report,'recorded':{k:v for k,v in report['recorded'].items() if k not in ['statistics','streaming_failures']}}
    brief['recorded']['streaming_failures_count']=len(report['recorded']['streaming_failures'])
    print(json.dumps(brief,indent=2))
if __name__=='__main__':main()
