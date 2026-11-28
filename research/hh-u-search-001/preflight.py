import argparse,json,math,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from safetensors.torch import load_file
from variants import ARMS,fresh,from_unified,tensor_bytes
from model import NativeLM,count
from screen import side_options,sha
R=Path(__file__).resolve().parent

def mask_forward(u,x,pattern):
    h,last,ax=u.local(x,u.core.emb)
    for i,c in enumerate(pattern):h,_=u.core.blocks[i](h,ax,u.payload_map[i],None,side_options(u.sides[i],h)[c])
    return F.linear(u.core.norm(h),u.core.emb.weight)
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert sha(R/'model.py')=='eaf5ae7973957bc68330fe754a8aa6cdb1199189413fee6995aa749f809e75c8'
    assert sha(a.data/'manifest.json')=='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35';r=json.loads((a.parent/'model_receipt.json').read_text());assert r['lock']['arm']=='U' and r['targets']==100663296 and sha(a.parent/'model.safetensors')==r['sha256']
    u=NativeLM('U');u.load_state_dict(load_file(str(a.parent/'model.safetensors')));u.eval();raw=np.memmap(a.data/'train.bin',dtype=np.uint8,mode='r');x=torch.from_numpy(np.array(raw[4096:4096+2*65],dtype=np.int64).reshape(2,65));rows=[]
    for arm in ARMS:
        q=from_unified(u,arm).eval();m=fresh(arm,9301);m.eval()
        with torch.inference_mode():
            z=q(x[:,:64]);zz=mask_forward(u,x[:,:64],arm);err=float((z-zz).abs().max());assert err<2e-5,(arm,err)
            z,st=m(x[:,:64],carry=True);z1,s1=m(x[:,:19],carry=True);z2,s2=m(x[:,19:64],state=s1,carry=True);chunk=float((z-torch.cat([z1,z2],1)).abs().max());assert chunk<3e-5,(arm,chunk)
            alt=x[:,:64].clone();alt[:,32:]=(alt[:,32:]+53)%256;causal=float((m(alt)[:,:32]-z[:,:32]).abs().max());assert causal<3e-5,(arm,causal)
            state=tensor_bytes(st)//2;assert abs(count(m)-1077504)/1077504<.002
        m.train();o=torch.optim.AdamW(m.parameters(),lr=.00015);o.zero_grad();loss=F.cross_entropy(m(x[:,:64]).flatten(0,1),x[:,1:].flatten());loss.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(g);o.step();assert all(torch.isfinite(p).all() for p in m.parameters())
        rows.append(dict(arm=arm,parameters=count(m),ff=list(m.ff),state_bytes_per_example=state,mask_equivalence_max_error=err,chunk_max_error=chunk,causal_max_error=causal,finite_update=True));print(rows[-1],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='ENGINEERING_ONLY',scientific_fits=0,rows=rows),indent=2))
if __name__=='__main__':main()
