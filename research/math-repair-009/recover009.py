"""Validate reused corpus and terminal checkpoints; wait for original non-GPT stage1 jobs."""
import argparse,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path

ORIGINAL_RUN=46039004295
ARMS=('HH','A0','U','UA','AU')
SEEDS=(9001,9011,9029)
FILES={
'dev.bin':'7d5377e9aac6670c5cad00a3c5bfb0c76befdd36ae97c001e8a99442bf02337b',
'dev_documents.json':'7f78fe231653831b7e4b18d7d35ba79062f289a907d591884e2cedb4528ef3f4',
'dev_offsets.npy':'35ac552ef7ebb26f7ef9561f3d325b1edb3e31d0687c23ee7be9fe99a8067112',
'dev_starts.npy':'e41d0c8788e077fd718d8a58ad3976721ce8914520e735c5c4a009a6bdd471a3',
'train.bin':'65b720b7b429953c5ac9b79b8c3d426c0ceeac72ab01fe4cf678cfb99efe152b',
'train_documents.json':'93c40dc8d141899e6cc0bff68f3d801a0658133c8d007bfa29021b99ee0e3a83',
'train_offsets.npy':'002a6867887e78aeeb2fa568cfef708262a242d31472f0bc75d2836fd7dd7a49',
'train_starts.npy':'69d8ce9b74f9df5c25b5049af2a3782260e2f6a8f0f74b4263281713cd59b648'}
BASE={'model.py':'eaf5ae7973957bc68330fe754a8aa6cdb1199189413fee6995aa749f809e75c8',
'data009.py':'d0bf2a6e560487f254d4a14cb7887f30559c592f61010aa4e48cf1c5f091d27d',
'train009.py':'8037dd96af919c14e7b17a6f8a608c49fbe1d1cb48274b337cf08990434156d7',
'PROTOCOL.md':'7fe339597c3e2e6f95d1748dad8e4d64dc8ed0b9d7c5af8cf576dda1bc18211b'}
HERE=Path(__file__).resolve().parent

def source(arm):
    from data009 import sha
    assert all(sha(HERE/k)==v for k,v in BASE.items()),'scientific source changed'
    return dict(BASE,**({'gpt009.py':sha(HERE/'gpt009.py')} if arm=='gpt' else {}))

def corpus(root):
    from data009 import verify,sha
    root=Path(root);m=verify(root)
    assert m['split']=='train' and m['files']==FILES
    assert m['train']==dict(documents=28720,bytes=529079832,eligible_windows=674661)
    assert m['dev']==dict(documents=282,bytes=5029466,eligible_windows=6411)
    assert [x['sha256'] for x in m['sources']]==['eda855422d336638fe4507fe7b7a7961f3f7b821a93306bab7b43578e243a2ca','75aa65dee9de2a7c10ba1808efd2408c3f4eb008104c3ccac47f8ed19300ebdd']
    print(json.dumps(dict(status='ORIGINAL_CORPUS_EXACT',manifest_sha256=sha(root/'manifest.json'),files=FILES)),flush=True)
    return m

def await_native(out):
    token=os.environ['GH_TOKEN'];repo=os.environ['GITHUB_REPOSITORY'];assert repo=='FPC-effortless/cdl-attention-experiment'
    url=f'https://api.github.com/repos/{repo}/actions/runs/{ORIGINAL_RUN}/jobs?per_page=100'
    expected={f'stage1 ({a}, {s})' for a in ARMS for s in SEEDS}
    deadline=time.monotonic()+5*3600
    while True:
        try:
            req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
            with urllib.request.urlopen(req,timeout=45) as response:page=json.load(response)
            assert page['total_count']<=100,'pagination required; do not assume completeness'
            jobs={j['name']:j for j in page['jobs'] if j['name'] in expected}
            assert set(jobs)==expected,'original matrix identity mismatch'
            failed=[j for j in jobs.values() if j['status']=='completed' and j['conclusion']!='success']
            if failed:raise RuntimeError('Original native stage failed: '+json.dumps([{k:j.get(k) for k in ('id','name','status','conclusion')} for j in failed]))
            good=sum(j['status']=='completed' and j['conclusion']=='success' for j in jobs.values())
            print(json.dumps(dict(original_run=ORIGINAL_RUN,terminal_native_jobs=good,expected=len(expected))),flush=True)
            if good==len(expected):
                p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(run=ORIGINAL_RUN,jobs=[{k:j.get(k) for k in ('id','name','status','conclusion','completed_at')} for j in jobs.values()]),indent=2));return
        except (urllib.error.URLError,TimeoutError) as e:
            print('Transient original-run status read failed:',type(e).__name__,flush=True)
        if time.monotonic()>=deadline:raise TimeoutError('Original native stages not all complete; retry this barrier, not the training fits.')
        time.sleep(60)

def checkpoint(a):
    from data009 import T,B,STEPS,STAGE,sha,order
    import numpy as np,torch
    corpus(a.data)
    p=Path(a.checkpoint);r=json.loads(p.with_name('receipt.json').read_text());step=a.stage*STAGE
    assert r['terminal_stage'] is True and r['step']==step and r['targets']==step*B*T and str(r['run_id'])==str(a.run_id)
    assert sha(p)==r['sha256']
    cp=torch.load(p,weights_only=True,map_location='cpu');assert cp['step']==step and cp['lock']==r['lock']
    l=cp['lock'];assert l['arm']==a.arm and l['seed']==a.seed and l['source']==source(a.arm)
    assert l['data']==sha(Path(a.data)/'manifest.json') and l['data_order']==hashlib.sha256(order(a.data,a.seed).tobytes()).hexdigest()
    assert l['steps']==STEPS and l['context']==T and l['batch']==B and l['threads']==4
    assert l['torch']==str(torch.__version__) and l['numpy']==np.__version__
    assert cp['optimizer']['state'] and cp['weights'] and torch.is_tensor(cp['rng'])
    assert cp['curve'][-1]['step']==step and 'dev_bpb' in cp['curve'][-1]
    assert p.with_name('development.json').is_file()
    print(json.dumps(dict(status='TERMINAL_CHECKPOINT_VERIFIED',run=a.run_id,arm=a.arm,seed=a.seed,step=step,sha256=r['sha256'],targets=r['targets'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    c=s.add_parser('corpus');c.add_argument('--data',required=True)
    w=s.add_parser('await-native');w.add_argument('--out',required=True)
    c=s.add_parser('checkpoint');c.add_argument('--checkpoint',required=True);c.add_argument('--data',required=True);c.add_argument('--arm',choices=('gpt',)+ARMS,required=True);c.add_argument('--seed',type=int,choices=SEEDS,required=True);c.add_argument('--stage',type=int,choices=(1,2,3),required=True);c.add_argument('--run-id',required=True)
    a=p.parse_args()
    if a.mode=='corpus':corpus(a.data)
    elif a.mode=='await-native':await_native(a.out)
    else:checkpoint(a)
