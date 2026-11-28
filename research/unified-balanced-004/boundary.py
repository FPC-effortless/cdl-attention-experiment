"""Matched training-objective extension for whole-model COG. No parent edits."""
import argparse,hashlib,json,sys,statistics
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
from space import BASE,BASE_ARMS,WholeCOG
from balanced import generate,draw8_balanced,independent_answer,counterfactuals
from knowledge import split,predict,VAL,KEY,WRITE,RESET,HOLD
import train as t
HERE=Path(__file__).resolve().parent
ANCHORS=('NN_native_native','SS_presence_factor','BB_presence_factor','GQA')
FILES=('space.py','balanced.py','boundary.py','PROTOCOL.md','balanced_preflight.py','verify_balanced.py','audit_objective.py')
PARENT_RUN='46124093141'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source():return dict(t.source(),**{'balanced004/'+n:sha(HERE/n) for n in FILES})
def read_fit(folder,new):
    p=Path(folder);r=json.loads((p/'result.json').read_text());c=r['config'];arm,seed=c['arm'],c['seed']
    assert arm in ANCHORS and seed in t.SEEDS and tuple(c['recipe'])==BASE_ARMS[arm]
    assert r['status']=='TERMINAL_DEVELOPMENT_ONLY' and r['terminal'] and r['reload_verified']
    assert c['steps']==1024 and c['targets']==12582912 and c['examples4']==c['examples8']==8192 and c['corpus']==t.CORPUS
    assert c['source']==(source() if new else t.source())
    if not new:assert str(c['run_id'])==PARENT_RUN
    for fn,k in [('checkpoint.pt','checkpoint_sha256'),('model.safetensors','model_sha256'),('metrics.json','metrics_sha256')]:assert sha(p/fn)==r[k],(arm,seed,fn)
    rec=json.loads((p/'receipt.json').read_text());assert rec['terminal'] and rec['step']==1024 and rec['sha256']==r['checkpoint_sha256'] and rec['config']==c
    m=json.loads((p/'metrics.json').read_text());assert m['text']['targets']==5029184 and m['text']['bpb']==r['language_bpb'] and m['cognition4']==r['cognition4'] and m['cognition8']==r['cognition8']
    return r
@torch.no_grad()
def panel(folder,new):
    folder=Path(folder);r=read_fit(folder,new);c=r['config'];arm,seed=c['arm'],c['seed'];m=WholeCOG(arm,seed);w=load_file(str(folder/'model.safetensors'));m.load_state_dict(w,strict=True);m.eval()
    # Exact parent evaluation is re-executed, not taken solely from its log.
    assert t.eval4(m,seed)==r['cognition4'] and t.eval8(m,seed)==r['cognition8'],(arm,seed,'parent evaluator mismatch')
    _,dev,_=split(seed,8);x,y,meta=generate(np.random.default_rng(seed+8601),dev,256,True);assert torch.equal(independent_answer(x),y)
    ix,x2,y2,eligible=counterfactuals(x,y,meta,dev);p=predict(m,x);p2=predict(m,x2)
    direct=torch.from_numpy(((meta['change'][:,0]==WRITE)|(meta['change'][:,0]==RESET))&(meta['change'][:,1]==meta['q']+KEY));ok=p==y
    def stats(mask):return dict(n=int(mask.sum()),correct=int((ok&mask).sum()),valid=int((((p>=VAL)&(p<=VAL+4))&mask).sum()))
    groups={str(j):stats(torch.from_numpy(meta['bucket']==j)) for j in range(8)}
    result=dict(arm=arm,seed=seed,regime='balanced' if new else 'original',checkpoint_sha256=r['checkpoint_sha256'],model_sha256=r['model_sha256'],parent_replay=True,source=source(),panel_input_sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()+ix.tobytes()+x2.numpy().tobytes()+y2.numpy().tobytes()).hexdigest(),overall=stats(torch.ones(len(x),dtype=torch.bool)),direct=stats(direct),prefix=stats(~direct),buckets=groups,paired=dict(eligible=eligible,n=len(ix),both_correct=int(((p[ix]==y[ix])&(p2==y2)).sum()),changed=int((p[ix]!=p2).sum())),predictions=p.tolist(),targets=y.tolist(),paired_indices=ix.tolist(),paired_predictions=p2.tolist(),paired_targets=y2.tolist())
    assert t.same(w,m.state_dict()) and sha(folder/'model.safetensors')==r['model_sha256']
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('train','panel','compare'));p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--fit',type=Path);p.add_argument('--original',type=Path);p.add_argument('--new',type=Path);p.add_argument('--arm',choices=ANCHORS);p.add_argument('--seed',type=int,choices=t.SEEDS);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    if a.mode=='train':
        assert a.arm and a.seed and a.data
        # Reuse exact trainer. Only replace eight-key data distribution and extend source lock.
        original_source=t.source
        def locked():return dict(original_source(),**{'balanced004/'+n:sha(HERE/n) for n in FILES})
        t.source=locked;t.draw8=draw8_balanced
        sys.argv=[str(BASE/'train.py'),'--arm',a.arm,'--seed',str(a.seed),'--data',str(a.data),'--out',str(a.out),'--threads',str(a.threads)]
        t.main();t.source=original_source
        z=panel(a.out,True);(a.out/'balanced_panel.json').write_text(json.dumps(z,indent=2)+'\n');print('BALANCED_PANEL',a.arm,a.seed,z['overall'],z['prefix'],z['paired'],flush=True);return
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.mode=='panel':
        assert a.fit;a.out.parent.mkdir(parents=True,exist_ok=True);z=panel(a.fit,False);a.out.write_text(json.dumps(z,indent=2)+'\n');print({k:z[k] for k in ('arm','seed','overall','direct','prefix','paired','parent_replay')},flush=True);return
    assert a.original and a.new
    expected={(arm,s) for arm in ANCHORS for s in t.SEEDS};original={};new={};panels={};rows=[]
    for root,flag,d in ((a.original,False,original),(a.new,True,new)):
        for path in root.rglob('result.json'):
            raw=json.loads(path.read_text());c=raw['config'];k=(c['arm'],c['seed'])
            if k not in expected:continue
            assert k not in d;d[k]=read_fit(path.parent,flag)
            z=panel(path.parent,flag);panels[(k,flag)]=z
            if flag:assert json.loads((path.parent/'balanced_panel.json').read_text())==z
    assert set(new)==expected,('new fits missing',expected-set(new))
    for seed in t.SEEDS:
        for group in (new,original):
            v=[r for k,r in group.items() if k[1]==seed]
            if v:assert len({r['config']['text_order_sha256'] for r in v})==1 and len({json.dumps(r['config']['knowledge_splits'],sort_keys=True) for r in v})==1 and len({r['knowledge_draw_sha256'] for r in v})==1
    def metrics(r,z):
        return dict(language_bpb=r['language_bpb'],four_full=r['cognition4']['full']['correct']/256,four_paired=r['cognition4']['full']['both_correct']/r['cognition4']['full']['pairs'],eight_held=r['cognition8']['held_full']['correct']/256,eight_paired=r['cognition8']['held_full']['both_correct']/r['cognition8']['held_full']['pairs'],eight_long=r['cognition8']['held_long']['correct']/256,revision=r['cognition8']['revision']['both_correct']/r['cognition8']['revision']['pairs'],balanced_prefix=z['prefix']['correct']/z['prefix']['n'],balanced_paired=z['paired']['both_correct']/z['paired']['n'],balanced_direct=z['direct']['correct']/z['direct']['n'],state_bytes=r['state_bytes_13'])
    for arm,seed in sorted(expected):
        k=(arm,seed);r=new[k];z=panels[k,True];v=dict(arm=arm,seed=seed,new=metrics(r,z),new_checkpoint=r['checkpoint_sha256'])
        if k in original:
            old=original[k];q=panels[k,False];assert old['config']['text_order_sha256']==r['config']['text_order_sha256'] and old['config']['knowledge_splits']==r['config']['knowledge_splits'] and old['config']['parameters']==r['config']['parameters'] and q['panel_input_sha256']==z['panel_input_sha256'];v['original']=metrics(old,q);v['delta']={key:v['new'][key]-val for key,val in v['original'].items()};v['original_checkpoint']=old['checkpoint_sha256']
        rows.append(v)
    complete=set(original)==expected;summary=dict(status='PAIRED_DEVELOPMENT_ONLY' if complete else 'WAITING_FOR_ORIGINAL_CONTROLS',new_terminal_fits=len(new),original_reused_fits=len(original),missing_original=[list(k) for k in sorted(expected-set(original))],rows=rows,warning='No promotion. Same training budget, different eight-key distribution. Paired language/cognition effects are development evidence; sealed tests and full-budget confirmation still required.')
    if complete:summary['paired_mean_deltas']={arm:{key:statistics.mean(v['delta'][key] for v in rows if v['arm']==arm) for key in rows[0]['delta']} for arm in ANCHORS}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,indent=2)+'\n');print(a.out.read_text(),flush=True)
if __name__=='__main__':main()
