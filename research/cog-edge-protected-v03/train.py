"""Matched protected-state simplification, developmental battery telemetry only."""
import argparse,copy,hashlib,json,math,pathlib,random,sys,time
import numpy as np
import torch
from torch import nn

def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def fit(edge,protected,name,seed,tr,selection,mean,std,path):
    torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    m=protected.make(name);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.002)
    path.mkdir(parents=True,exist_ok=False);torch.save(m.state_dict(),path/'initial.pt')
    start=time.perf_counter();trace=[];best=None;stream=hashlib.sha256()
    def update(record=False):
        m.train();x,t,s=edge.batch(tr,rng)
        if record:
            for v in (x,t,s):stream.update(v.numpy().tobytes())
        u,v=m(x);u=u[:,:t.shape[1]];v=v[:,:t.shape[1]]
        loss=(.5*((u-t)/v).square()+v.log())[s].mean();opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        value=float(loss.detach());assert math.isfinite(value);return value
    def snap(i):
        return {'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'torch_rng':torch.get_rng_state().clone(),'numpy_rng':copy.deepcopy(rng.bit_generator.state),'python_rng':random.getstate(),'update':i,'mean':mean.tolist(),'std':std.tolist(),'name':name,'seed':seed}
    for i in range(600):
        loss=update(True)
        if (i+1)%120==0:
            val=edge.val(edge.predict(m,selection),std[0]);assert math.isfinite(val);trace.append([i+1,loss,val])
            if best is None or val<best[0]:best=(val,snap(i+1))
    train_s=time.perf_counter()-start
    torch.save(snap(600),path/'final.pt');torch.save(best[1],path/'selected.pt')
    loss=update();state=copy.deepcopy(m.state_dict());ostate=copy.deepcopy(opt.state_dict())
    c=torch.load(path/'final.pt',weights_only=True);m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['torch_rng']);rng.bit_generator.state=c['numpy_rng'];random.setstate(c['python_rng'])
    assert loss==update() and all(torch.equal(state[k],v) for k,v in m.state_dict().items())
    for k,p in opt.state_dict()['state'].items():
        assert all(torch.equal(v,ostate['state'][k][n]) if isinstance(v,torch.Tensor) else v==ostate['state'][k][n] for n,v in p.items())
    return {'name':name,'seed':seed,'params':sum(p.numel() for p in m.parameters()),'raw_recurrent_bytes':{'cog':96,'protected-instant':64,'dense-instant':64,'protected-compact':64,'static':0}[name],'selected':best[1]['update'],'trace':trace,'val':best[0],'training_s_including_selection_and_hashing':train_s,'training_stream_sha256':stream.hexdigest(),'resume_exact':True,'checkpoint_sha256':{p.name:sha(p.read_bytes()) for p in path.glob('*.pt')}}

def main():
    p=argparse.ArgumentParser()
    for n in ('restored','dataset','archived','fold','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();sys.path.insert(0,a.restored);import edge,protected
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert a.fold in ('Y','R','G')
    root=pathlib.Path(a.archived);path=pathlib.Path(a.dataset);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    raw=(root/'source-manifest.json').read_bytes();assert raw==path.with_suffix('.json').read_bytes()
    meta=json.loads(raw);source={x['key']:x for x in meta['source']};old=meta['splits'][a.fold]
    data=np.load(path,allow_pickle=False);assert len(source)==215 and set(data.files)==set(source)
    for k in data.files:assert len(data[k])==source[k]['seconds'] and sha(data[k].astype('<f4').tobytes())==source[k]['array_sha256']
    split=json.loads(pathlib.Path(__file__).with_name('splits.json').read_text())[a.fold]
    assert split['train']==old['train'] and split['test']==old['test']
    assert set(split['selection'])|set(split['calibration'])==set(old['dev'])
    packs=[{source[n]['pack'] for n in v} for v in split.values()]
    assert all(not packs[i]&packs[j] for i in range(4) for j in range(i))
    # The historical "calibration" subset is the developmental comparison set here.
    # Do not construct or score the old held-out-aircraft test panels.
    rawtr=np.concatenate([data[n].astype(np.float32) for n in split['train']]);mean=rawtr.mean(0);std=rawtr.std(0)
    norm=lambda ids:[(data[n].astype(np.float32)-mean)/std for n in ids]
    tr=norm(split['train']);selection=edge.panels(norm(split['selection']),split['selection'],'dev-v1')
    result={'scope':'development-only; previously studied dataset; no old test scoring, no calibration/qualification claim','fold':a.fold,'source_manifest_sha256':sha(raw),'splits':split,'mean':mean.tolist(),'std':std.tolist(),'updates':600,'fits':[],'env':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__}}
    for seed in edge.SEEDS:
        same=[]
        for name in protected.ARMS:
            f=fit(edge,protected,name,seed,tr,selection,mean,std,out/f'{seed}-{name}')
            result['fits'].append(f);same.append(f['training_stream_sha256']);write(out/'results.json',result)
            print('fit',a.fold,name,seed,f['params'],f['selected'],flush=True)
        assert len(set(same))==1
        shapes=[torch.load(out/f'{seed}-{n}'/'initial.pt',weights_only=True) for n in ('cog','protected-instant','dense-instant')]
        assert all(set(x)==set(shapes[0]) for x in shapes)
        assert all(torch.equal(shapes[0][k],x[k]) for x in shapes[1:] for k in shapes[0])
    compare=edge.panels(norm(split['calibration']),split['calibration'],'cal-v2')
    target_counts=None
    for f in result['fits']:
        m=protected.make(f['name']);m.load_state_dict(torch.load(out/f"{f['seed']}-{f['name']}"/'selected.pt',weights_only=True)['model']);m.eval()
        metrics,rows=edge.metrics(edge.predict(m,compare),std[0])
        counts={(x['flight'],x['regime'],h):x[str(h)]['n'] for x in rows for h in edge.H}
        if target_counts is None:target_counts=counts
        assert counts==target_counts
        rr=[protected.checks(m,x,f['name']) for _,_,x,*_ in compare]
        f['development_metrics']=metrics;f['flight_metrics']=rows
        f['checks']={'protected_checks':sum(x['protected_checks'] for x in rr),'max_step_delta':max((x['step_delta'] for x in rr if x['step_delta'] is not None),default=None),'max_prefix_delta':max(x['prefix_delta'] for x in rr),'exact_restarts':sum(x['restart_exact'] is True for x in rr)}
        write(out/'results.json',result)
        print('evaluated',a.fold,f['name'],f['seed'],metrics['0.15-1/15']['rmse_v'],flush=True)
    result['complete']=True;write(out/'results.json',result)

if __name__=='__main__':main()
