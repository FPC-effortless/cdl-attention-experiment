import torch, json, pathlib, time, math
from scan_reference import scan
P=pathlib.Path(__file__).parent

def assoc(z,a,s=None):
 if s is None:s=torch.zeros_like(z[:,0])
 yy=[]
 for zc,ac in zip(z.split(16,1),a.split(16,1)):
  A=ac; B=(1-ac)*zc
  offset=1
  while offset<zc.shape[1]:
   A,B=torch.cat((A[:,:offset],A[:,offset:]*A[:,:-offset]),1),torch.cat((B[:,:offset],B[:,offset:]+A[:,offset:]*B[:,:-offset]),1)
   offset*=2
  y=A*s[:,None]+B;s=y[:,-1];yy.append(y)
 return torch.cat(yy,1),s

def seq(z,a,s=None):
 if s is None:s=torch.zeros_like(z[:,0])
 out=[]
 for i in range(z.shape[1]):s=a[:,i]*s+(1-a[:,i])*z[:,i];out.append(s)
 return torch.stack(out,1),s

def diff(v,ref):
 return dict(finite=bool(v.isfinite().all()),max_error=float((v.double()-ref).abs().max()) if v.isfinite().all() else None)
def main():
 torch.set_num_threads(1);torch.manual_seed(89012);res=[];gr=[];special=[]
 for dt in [torch.float32,torch.float16,torch.bfloat16]:
  for d in [48,96,192]:
   for L in [32,128,1024]:
    z=(torch.rand(2,L,d)*2-1).to(dt);a=(.5+.499*torch.rand_like(z.float())).to(dt);s=(torch.rand(2,d)*2-1).to(dt);ref,_=seq(z.double(),a.double(),s.double());r=dict(dtype=str(dt),d=d,L=L)
    for name,f in [('division',scan),('associative',assoc),('sequential',seq)]:
     v,_=f(z,a,s);r[name]=diff(v,ref)
    res.append(r)
  for sign in [-1,1]:
   z=torch.full((1,32,48),float(sign),dtype=dt);a=torch.full_like(z,.5);ref,_=seq(z.double(),a.double());r=dict(dtype=str(dt),sign=sign)
   for name,f in [('division',scan),('associative',assoc)]:r[name]=diff(f(z,a)[0],ref)
   special.append(r)
 for d in [48,96,192]:
  for L in [32,128]:
   zz=torch.rand(2,L,d)*2-1;aa=.5+.49*torch.rand_like(zz);ss=torch.rand(2,d)*2-1;w=torch.randn_like(zz);gs={}
   for name,f in [('sequential',seq),('division',scan),('associative',assoc)]:
    z,a,s=[v.clone().requires_grad_() for v in [zz,aa,ss]];v,_=f(z,a,s);g=torch.autograd.grad((v*w).mean(),[z,a,s]);gs[name]=g
   gr.append(dict(d=d,L=L,**{name:[float((u-v).abs().max()) for u,v in zip(gs[name],gs['sequential'])] for name in ['division','associative']}))
 result=dict(random=res,stress=special,gradient=gr,sensitivity={str(n):{'standard_bound':.99**n,'slow_bound':.999**n} for n in [32,128,256,1024,4096]});(P/'scan_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps({'stress':special,'gradient_max':{n:max(max(v[n]) for v in gr) for n in ['division','associative']},'random_max':{str(dt):{n:max(v[n]['max_error'] for v in res if v['dtype']==str(dt) and v[n]['finite']) for n in ['division','associative']} for dt in [torch.float32,torch.float16,torch.bfloat16]}}))
if __name__=='__main__':main()
