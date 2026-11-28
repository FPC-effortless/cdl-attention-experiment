import torch,json,pathlib
from scan import associative,sequential
from hierarchical import hierarchical
P=pathlib.Path(__file__).parent

def exact_hold_scan(z,a,s=None,fn=hierarchical):
 """For hard, externally selected HOLD gates. Value restoration is not a smooth gate derivative."""
 if s is None:s=torch.zeros_like(z[:,0],dtype=torch.float32)
 else:s=s.float()
 z,a=z.float(),a.float();y,_=fn(z,a,s);idx=torch.arange(1,z.shape[1]+1,device=z.device).reshape(1,-1,*([1]*(z.ndim-2)));idx=torch.where(a!=1,idx,0).cummax(1).values
 out=torch.gather(torch.cat([s[:,None],y],1),1,idx.expand_as(y));return out,out[:,-1]

if __name__=='__main__':
 torch.set_num_threads(1);out=[]
 for dtype in [torch.float32,torch.float16,torch.bfloat16]:
  for d in [192,320,576]:
   for L in [32,256,1024]:
    for seed in [173,271]:
     g=torch.Generator().manual_seed(seed+d+L);z=(2*torch.rand(2,L,d,generator=g)-1).to(dtype).float();a=(.5+.49*torch.rand(2,L,d,generator=g)).to(dtype).float();s=(2*torch.rand(2,d,generator=g)-1).to(dtype).float();u=torch.rand(2,L,d,generator=g);hold=u<.25;reset=u>.85;a[hold]=1;a[reset]=0;z2=z.clone();z2[hold]=-z2[hold];r={'dtype':str(dtype),'d':d,'L':L,'seed':seed,'hold_n':int(hold.sum()),'reset_n':int(reset.sum())};ref,_=sequential(z.double(),a.double(),s.double())
     for name,fn in [('assoc',associative),('hier',hierarchical)]:
      y,_=exact_hold_scan(z,a,s,fn);old=torch.cat([s[:,None],y[:,:-1]],1);yy,_=exact_hold_scan(z2,a,s,fn);r[name]={'hold_bits_exact':int((y[hold].view(torch.int32)==old[hold].view(torch.int32)).sum()),'reset_value_exact':int((y[reset]==z[reset]).sum()),'hold_perturb_equal':torch.equal(y,yy),'relative_l2':float((y.double()-ref).norm()/ref.norm().clamp_min(1e-8))}
     out.append(r)
 (P/'hold_repair.json').write_text(json.dumps(out,indent=2));print('panels',len(out));print('holds',sum(r['hold_n'] for r in out),'resets',sum(r['reset_n'] for r in out))
 for k in ['assoc','hier']:print(k,'hold_bits_mismatch',sum(r['hold_n']-r[k]['hold_bits_exact'] for r in out),'reset_mismatch',sum(r['reset_n']-r[k]['reset_value_exact'] for r in out),'relative_l2',max(r[k]['relative_l2'] for r in out))
