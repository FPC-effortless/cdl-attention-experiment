import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import load_file
from model import NativeLM
HERE=Path(__file__).resolve().parent
MASKS=('N','S','R','B')
SEEDS=(9001,9011,9029)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def side_options(side,h):
 features=[]
 handle=side.proj.register_forward_pre_hook(lambda mod,args:features.append(args[0]))
 try:full,_=side(h)
 finally:handle.remove()
 x=features[0];w=side.proj.weight
 return dict(N=None,S=F.linear(x[...,:8],w[:,:8]),R=F.linear(x[...,8:],w[:,8:]),B=full)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--threads',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--data',type=Path,default=HERE/'corpus');p.add_argument('--parents',type=Path,default=HERE/'parents');p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 assert sha(HERE/'model.py')=='eaf5ae7973957bc68330fe754a8aa6cdb1199189413fee6995aa749f809e75c8'
 root=a.data;assert sha(root/'manifest.json')=='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35';mfest=json.loads((root/'manifest.json').read_text());assert all(sha(root/n)==v for n,v in mfest['files'].items() if n.startswith('dev'))
 raw=np.memmap(root/'dev.bin',mode='r',dtype=np.uint8);off=np.load(root/'dev_offsets.npy',allow_pickle=False);xs=[];docs=[]
 for i,(l,r) in enumerate(zip(off[:-1],off[1:])):
  l,r=int(l),int(r)
  if r-l<641:continue
  pos=l+int(hashlib.sha256(f'hh-u-comb-001:{i}'.encode()).hexdigest(),16)%(r-l-641+1);xs.append(np.array(raw[pos:pos+641],dtype=np.int64));docs.append({'doc':i,'start':pos,'targets':128})
 partial=a.limit is not None;xs=np.stack(xs);eligible=len(xs);a.out.mkdir(parents=True,exist_ok=True);idx=sorted(range(eligible),key=lambda i:hashlib.sha256(f'hh-u-partial-001:{docs[i]["doc"]}'.encode()).hexdigest());
 if a.limit is not None:idx=idx[:a.limit];xs=xs[idx];docs=[docs[i] for i in idx]
 (a.out/'selection.json').write_text(json.dumps(docs,indent=2));print('documents',len(xs),'eligible',eligible,'partial',a.limit is not None,flush=True)
 for seed in (SEEDS if a.seed is None else (a.seed,)):
  out=a.out/f'{seed}.json';assert not out.exists(),out
  models={}
  for arm in ('HH','U'):
   parent=a.parents/f'{arm}-{seed}';r=json.loads((parent/'model_receipt.json').read_text());assert sha(parent/'model.safetensors')==r['sha256'] and r['targets']==100663296
   models[arm]=NativeLM(arm).eval();models[arm].load_state_dict(load_file(str(parent/'model.safetensors')),strict=True)
  u=models['U'];hh=models['HH'];rows={a+b:[] for a in MASKS for b in MASKS};rows['HH']=[];err=0.;t0=time.monotonic()
  def score(z,y,which):
   z=z[:,-128:];nll=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape_as(y).double().sum(1);ok=(z.argmax(-1)==y).sum(1);rows[which].extend([[float(a),int(b)] for a,b in zip(nll,ok)])
  with torch.inference_mode():
   for i in range(0,len(xs),16):
    x=torch.from_numpy(xs[i:i+16,:-1].copy());y=torch.from_numpy(xs[i:i+16,-128:].copy());h,_,ax=u.local(x,u.core.emb);first=side_options(u.sides[0],h)
    score(hh(x),y,'HH')
    for m1 in MASKS:
     h1,_=u.core.blocks[0](h,ax,u.payload_map[0],None,first[m1]);second=side_options(u.sides[1],h1)
     for b in MASKS:
      h2,_=u.core.blocks[1](h1,ax,u.payload_map[1],None,second[b]);z=F.linear(u.core.norm(h2),u.core.emb.weight)
      if m1==b=='B' and i==0:
       err=float((z-u(x)).abs().max());assert err==0.,err
      score(z,y,m1+b)
    if (i//16)%5==0:print(seed,'documents',min(i+16,len(xs)),'/',len(xs),'seconds',round(time.monotonic()-t0,2),flush=True)
  assert all(len(r)==len(xs) for r in rows.values());summary={k:sum(x[0] for x in r)/(len(xs)*128*math.log(2)) for k,r in rows.items()};out.write_text(json.dumps(dict(seed=seed,partial=partial,eligible_documents=eligible,documents=len(xs),targets_per_panel=len(xs)*128,source_sha256=sha(HERE/'model.py'),full_u_exact_error=err,bpb=summary,rows=rows,seconds=time.monotonic()-t0)));print(seed,summary,flush=True)
if __name__=='__main__':main()
