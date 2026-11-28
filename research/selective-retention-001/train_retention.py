"""Matched continued pretraining from exact REPAIR-009 terminal HH weights.
No official validation/test; 6 arms x 3 parent seeds, each 12,582,912 new targets.
"""
import argparse, hashlib as h,json,math,os,platform,resource,sys,time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from trainable_model import adapt
from model import count
R=Path(__file__).resolve().parent
ARMS={'native_short':('native',768),'native_long':('native',1536),'time_long':('time512',1536),'write_long':('write128',1536),'scalar_long':('learned_scalar',1536),'context_long':('learned_write',1536)}
SEEDS=(9001,9011,9029);STEPS=1024;B=8;L=1536;TARGETS=STEPS*B*L
PARENT_HASHES={9001:'4d532cb33b48166e57043ea22b1f799387da025d2a9be209d6083d7e5be0156e',9011:'ccc641258da539ae17a97a13e4556085be20b8a614fb723d5c97f63d2d6fdfc2',9029:'451d0f3a4f9ef7b59644a335bb22a5d2c51fc12aa999d99f34c53c36b7274337'}
SOURCE='eaf5ae7973957bc68330fe754a8aa6cdb1199189413fee6995aa749f809e75c8';CORPUS='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'
def sha(p):
 with Path(p).open('rb') as f:return h.file_digest(f,'sha256').hexdigest()
def save_json(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def source_lock():
 assert sha(R/'model.py')==SOURCE;return {n:sha(R/n) for n in ('model.py','retention.py','trainable_model.py','train_retention.py','TRAIN_PROTOCOL.md')}
def verify_data(root):
 assert sha(root/'manifest.json')==CORPUS;m=json.loads((root/'manifest.json').read_text());assert m['split']=='train' and m['revision']=='00aa25585682d4957f9e86edc73f59be7419af99';assert all(sha(root/k)==v for k,v in m['files'].items());return m

def order(root,seed):
 starts=np.load(root/'train_starts.npy',allow_pickle=False);off=np.load(root/'train_offsets.npy',allow_pickle=False)
 used=np.random.Generator(np.random.PCG64(seed+900090)).permutation(starts)[:8192*16].reshape(8192,16)
 anchors=np.concatenate([np.arange(a,b-L,L,dtype=np.int64) for a,b in zip(off[:-1],off[1:])]);assert np.isin(anchors,starts).all() and np.isin(anchors+768,starts).all()
 anchors=anchors[~np.isin(anchors,used)&~np.isin(anchors+768,used)];assert len(anchors)>=STEPS*B and len(np.unique(anchors))==len(anchors)
 chosen=np.random.Generator(np.random.PCG64(seed+1780701)).permutation(anchors)[:STEPS*B].reshape(STEPS,B)
 return chosen,h.sha256(used.tobytes()).hexdigest(),len(anchors)

def parent(prior,seed,old_order):
 r=json.loads((prior/'model_receipt.json').read_text());c=json.loads((prior/'receipt.json').read_text());l=r['lock'];assert l==c['lock']==json.loads((prior/'lock.json').read_text());assert l['arm']=='HH' and l['seed']==seed and l['source']['model.py']==SOURCE and l['data']==CORPUS and l['data_order']==old_order
 assert r['step']==c['step']==8192 and r['targets']==c['targets']==100663296 and c['terminal_stage'] and str(c['run_id'])=='46044686359' and r['parameters']==1077504
 assert sha(prior/'model.safetensors')==r['sha256']==PARENT_HASHES[seed] and sha(prior/'checkpoint.pt')==c['sha256'];return load_file(str(prior/'model.safetensors')),r

def batches(arr,context):
 if context==768:arr=torch.cat([arr[:,:769],arr[:,768:]],0);return arr.split(4)
 return arr.split(2)

def opt(m):return torch.optim.AdamW([{'params':[p for p in m.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in m.parameters() if p.ndim<2],'weight_decay':0.}],lr=.00015,betas=(.9,.95))
def update(m,o,arr,context,step):
 m.train();o.zero_grad(set_to_none=True);ls=0.
 for a in batches(arr,context):
  z=m(a[:,:-1]);l=F.cross_entropy(z.flatten(0,1),a[:,1:].flatten())/4;assert torch.isfinite(l);l.backward();ls+=float(l.detach())
 gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn)
 lr=.00015*(step+1)/32 if step<32 else .00015*(.1+.45*(1+math.cos(math.pi*(step-32)/(STEPS-1-32))))
 for g in o.param_groups:g['lr']=lr
 o.step();return ls,float(gn),lr

def evaluate(m,root):
 m.eval();raw=np.memmap(root/'dev.bin',dtype=np.uint8,mode='r');off=np.load(root/'dev_offsets.npy',allow_pickle=False);xs=[];docs=[];rows=[]
 for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
  a,b=int(a),int(b);n=2177
  if b-a<n:continue
  pos=a+int(h.sha256(f'retention-adapt-001:{i}'.encode()).hexdigest(),16)%(b-a-n+1);xs.append(np.array(raw[pos:pos+n],dtype=np.int64));docs.append((i,pos))
 xs=np.stack(xs);assert len(xs)==276
 with torch.inference_mode():
  for n in (64,768,2048):
   for i in range(0,len(xs),16):
    x=torch.from_numpy(xs[i:i+16,2048-n:-1].copy());y=torch.from_numpy(xs[i:i+16,2049:].copy());state=None
    for a in x[:,:-128].split(256,1):_,state=m(a,state=state,carry=True)
    z=m(x[:,-128:],state=state);l=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape(y.shape);ok=z.argmax(-1)==y
    for j in range(len(x)):rows.append(dict(history=n,doc=docs[i+j][0],start=docs[i+j][1],targets=128,nll=float(l[j].double().sum()),correct=int(ok[j].sum())))
  assert len(rows)==276*3
  # Full document-held dev language at original 768-byte evaluation reset horizon.
  sums=np.zeros((len(off)-1,3),dtype=np.float64);windows=[]
  for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
   for j in range(int(a),int(b)-1,768):windows.append((i,j,min(768,int(b)-1-j)))
  windows.sort(key=lambda x:x[2])
  for i in range(0,len(windows),8):
   items=windows[i:i+8];n=max(x[2] for x in items);x=np.zeros((len(items),n),np.int64);y=np.full((len(items),n),-100,np.int64)
   for k,(_,p,l) in enumerate(items):x[k,:l]=raw[p:p+l];y[k,:l]=raw[p+1:p+l+1]
   x,y=torch.from_numpy(x),torch.from_numpy(y);z=m(x);ll=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none',ignore_index=-100).reshape(y.shape);ok=(z.argmax(-1)==y)&(y!=-100)
   for k,(doc,_,l) in enumerate(items):sums[doc]+=[float(ll[k].double().sum()),int(ok[k].sum()),l]
  total=sums.sum(0);assert int(total[2])==sum(max(0,int(b-a)-1) for a,b in zip(off[:-1],off[1:]))
  result={'full_dev_bpb_768':float(total[0]/(total[2]*math.log(2))),'full_dev_targets':int(total[2]),'full_per_doc':sums.tolist(),'context_rows':rows}
  for n in (64,768,2048):
   rr=[r for r in rows if r['history']==n];result[f'bpb_{n}']=sum(r['nll'] for r in rr)/(len(rr)*128*math.log(2))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(a.seed);a.out.mkdir(parents=True,exist_ok=False)
 lock=source_lock();verify_data(a.data);starts,old_order,pool=order(a.data,a.seed);weights,receipt=parent(a.prior,a.seed,old_order);mode,context=ARMS[a.arm];m=adapt(weights,mode);assert count(m)==1077504;o=opt(m);raw=np.memmap(a.data/'train.bin',dtype=np.uint8,mode='r');idx=np.arange(L+1);np.save(a.out/'order.npy',starts)
 conf=dict(arm=a.arm,mode=mode,context=context,seed=a.seed,steps=STEPS,targets=TARGETS,paired_windows_per_step=B,source=lock,corpus=CORPUS,parent_sha256=receipt['sha256'],parent_order=old_order,order_sha256=h.sha256(starts.tobytes()).hexdigest(),unused_pair_pool=pool,torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),threads=a.threads,commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'));save_json(a.out/'config.json',conf)
 curve=[];t0=time.monotonic();training=0.
 def checkpoint(step,terminal=False):
  tmp=a.out/'checkpoint.tmp';p=a.out/'checkpoint.pt';torch.save(dict(weights=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state(),step=step,config=conf,curve=curve),tmp);os.replace(tmp,p);save_json(a.out/'receipt.json',dict(step=step,targets=step*B*L,terminal=terminal,sha256=sha(p),config=conf))
 for step in range(STEPS):
  t=time.monotonic();arr=torch.from_numpy(raw[starts[step,:,None]+idx].astype(np.int64));l,g,lr=update(m,o,arr,context,step);training+=time.monotonic()-t
  if (step+1)%64==0:
   item=dict(step=step+1,loss=l,grad=g,lr=lr,targets=(step+1)*B*L,seconds=training);curve.append(item);print(json.dumps(item),flush=True)
  if (step+1)%256==0:checkpoint(step+1)
  if time.monotonic()-t0>5.2*3600 and step+1<STEPS:checkpoint(step+1);raise RuntimeError('Partial budget retained, no terminal credit')
 dev=evaluate(m,a.data);save_json(a.out/'development.json',dev);checkpoint(STEPS,True)
 w={k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()};p=a.out/'model.safetensors';save_file(w,str(p));rr=load_file(str(p));assert all(torch.equal(v,rr[k]) for k,v in w.items())
 cp=torch.load(a.out/'checkpoint.pt',weights_only=True,map_location='cpu');assert cp['step']==STEPS and cp['config']==conf and all(torch.equal(w[k],v) for k,v in cp['weights'].items());m2=adapt(weights,mode);m2.load_state_dict(cp['weights']);o2=opt(m2);o2.load_state_dict(cp['optimizer']);
 result=dict(config=conf,parameters=count(m),targets=TARGETS,terminal=True,checkpoint_sha256=sha(a.out/'checkpoint.pt'),model_sha256=sha(p),development_sha256=sha(a.out/'development.json'),training_seconds=training,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,reload_verified=True,**{k:v for k,v in dev.items() if k.startswith('bpb_') or k in ('full_dev_bpb_768','full_dev_targets')});save_json(a.out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
