"""Offline equipment-monitoring component qualification, historical pilot only."""
import argparse, hashlib, io, json, math, pathlib, statistics, sys, time, platform
import numpy as np
import torch
from torch import nn

def digest(b): return hashlib.sha256(b).hexdigest()

def guard(raw, mask):
    """Validate reported battery sensor observations; missing is not a fault."""
    raw=np.asarray(raw,dtype=np.float32); mask=np.asarray(mask,dtype=bool)
    if raw.shape!=(2,) or mask.shape!=(2,): raise ValueError("expected two channels")
    bad=mask & ~np.isfinite(raw)
    bad[0] |= bool(mask[0] and raw[0]<=0)
    return mask & ~bad, bad

def preprocess(norm, mask, last, age):
    last=np.where(mask,norm,last).astype(np.float32)
    age=np.where(mask,0,age+1).astype(np.float32)
    x=np.concatenate([np.where(mask,norm,0),mask,np.log1p(age)/4,last])
    return torch.tensor(x[None],dtype=torch.float32),last,age

def sparse_step(m,x,state):
    # Single-stream eager CPU reference, skips only known inactive sensor banks.
    h,p=state; mask=x[0,2:4].bool(); h=h.clone()
    active=mask.nonzero(as_tuple=True)[0]
    if len(active):
        z=torch.cat((x.expand(len(active),8),m.ids[active]),-1)
        g,u=m.enc(z).chunk(2,-1);g=g.sigmoid()
        h[0,active]=(1-g)*h[0,active]+g*u.tanh()
    v,w=m.dyn(x).chunk(2,-1);v=v.sigmoid()
    if m.mode=="gated":v=v*x[...,2:4].amax(-1,keepdim=True)
    p=(1-v)*p+v*w.tanh()
    return m.emit(h,p,x),(h,p)

def save_state(state,last,age,index,fingerprint):
    b=io.BytesIO();torch.save({"h":state[0],"p":state[1],"last":torch.tensor(last),"age":torch.tensor(age),"index":index,"fingerprint":fingerprint},b)
    raw=b.getvalue();return raw,digest(raw)

def load_state(raw,expected_sha,fingerprint):
    if digest(raw)!=expected_sha: raise ValueError("snapshot integrity failure")
    d=torch.load(io.BytesIO(raw),weights_only=True)
    if d["fingerprint"]!=fingerprint:raise ValueError("checkpoint identity mismatch")
    if d["h"].shape!=(1,2,8) or d["p"].shape!=(1,8) or d["last"].shape!=(2,) or d["age"].shape!=(2,):raise ValueError("shape mismatch")
    if not all(torch.isfinite(d[k]).all() for k in ("h","p","last","age")):raise ValueError("nonfinite snapshot")
    return (d["h"],d["p"]),d["last"].numpy(),d["age"].numpy(),d["index"]

def record(y,x,m,fp):
    # Compare normal preprocessing, masked step, and true conditional execution.
    s=m.init(1);ss=m.init(1);last=np.zeros(2,np.float32);age=np.full(2,32,np.float32)
    mask=x[0,:,2:4].numpy().astype(bool); full=[]; sparse=[];checks=0;max_x=0.;active_count=0
    midpoint=len(y)//2
    for j in range(len(y)):
        xx,last,age=preprocess(y[j],mask[j],last,age)
        max_x=max(max_x,float((xx-x[:,j]).abs().max()))
        o,s=m.step(x[:,j],s);old=ss;oo,ss=sparse_step(m,x[:,j],ss)
        assert torch.equal(old[0][0,~torch.from_numpy(mask[j])],ss[0][0,~torch.from_numpy(mask[j])])
        checks+=int((~mask[j]).sum())*8;active_count+=int(mask[j].sum())
        full.append(torch.cat(o,-1));sparse.append(torch.cat(oo,-1))
        if j==midpoint-1: raw,hashval=save_state(ss,last,age,j+1,fp)
    reference=torch.stack(full,1);candidate=torch.stack(sparse,1)
    diff=float((reference-candidate).abs().max());batch_diff=float((torch.cat(m(x),-1)-candidate).abs().max())
    assert max_x<2e-6 and diff<2e-5 and batch_diff<2e-5
    resumed,last,age,j0=load_state(raw,hashval,fp);out=[]
    assert j0==midpoint
    for j in range(j0,len(y)):
        xx,last,age=preprocess(y[j],mask[j],last,age);o,resumed=sparse_step(m,xx,resumed);out.append(torch.cat(o,-1))
    # Preprocessing parity can be float-close; exact restore with identical inputs is tested separately.
    restart_delta=float((torch.stack(out,1)-candidate[:,j0:]).abs().max());assert restart_delta<2e-5
    resumed,_,_,_=load_state(raw,hashval,fp);out=[]
    for j in range(j0,len(y)):
        o,resumed=sparse_step(m,x[:,j],resumed);out.append(torch.cat(o,-1))
    assert torch.equal(torch.stack(out,1),candidate[:,j0:])
    rejects=0
    for b,h,f in ((raw+b"x",hashval,fp),(raw,hashval,"other-model")):
        try:load_state(b,h,f)
        except ValueError: rejects+=1
    assert rejects==2
    return {"protected_coordinates":checks,"active_bank_updates":active_count,"possible_bank_updates":2*len(y),"step_delta":diff,"batch_delta":batch_diff,"preprocessing_delta":max_x,"restart_delta":restart_delta,"snapshot_bytes":len(raw),"corrupt_or_wrong_identity_rejections":rejects}

def bench(m,x):
    def run(fn):
        s=m.init(1);start=time.perf_counter_ns()
        for j in range(x.shape[1]):_,s=fn(x[:,j],s)
        return (time.perf_counter_ns()-start)/x.shape[1]/1000
    f=m.step;g=lambda xx,ss:sparse_step(m,xx,ss)
    for _ in range(2):run(f);run(g)
    a=[];b=[]
    for i in range(7):
        if i%2:b.append(run(g));a.append(run(f))
        else:a.append(run(f));b.append(run(g))
    return {"masked_us_samples":a,"conditional_us_samples":b,"median_ratio":statistics.median(a)/statistics.median(b)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--restored",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    root=pathlib.Path(a.restored);sys.path.insert(0,str(root));import edge
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    z=np.load(root/"pilot.npz",allow_pickle=False);meta=json.loads((root/"pilot.json").read_text())
    rows=[];timings=[];valid=0;faults=0;false_flags=0;missing_flags=0
    for y in z.values():
        for sample in y:
            ok,bad=guard(sample,[1,1]);valid+=2;false_flags+=int(bad.sum())
            for i,v in ((0,float("nan")),(1,float("inf")),(0,-1.)):
                corrupt=sample.copy();corrupt[i]=v;ok,bad=guard(corrupt,[1,1]);assert bad[i] and not ok[i] and ok[1-i];faults+=1
                ok,bad=guard(corrupt,[i!=0,i!=1]);missing_flags+=int(bad.sum())
    assert false_flags==0 and missing_flags==0
    with torch.inference_mode():
        for fold in "YRG":
            names=meta["splits"][fold]["test"]
            for seed in edge.SEEDS:
                for mode in ("cog","gated"):
                    ck=root/f"pilot-{fold}"/f"{seed}-{mode}"/"selected.pt";fp=digest(ck.read_bytes())
                    c=torch.load(ck,weights_only=True);m=edge.make(mode);m.load_state_dict(c["model"]);m.eval()
                    mean=np.asarray(c["mean"],np.float32);std=np.asarray(c["std"],np.float32)
                    ys=[(z[n]-mean)/std for n in names]
                    for name,reg,x,*_ in edge.panels(ys,names,"test-v1"):
                        y=ys[names.index(name)];r=record(y,x,m,fp);r.update(fold=fold,seed=seed,mode=mode,flight=name,regime=reg,checkpoint_sha256=fp);rows.append(r)
                        if mode=="cog" and name==names[0]:
                            t=bench(m,x);t.update(fold=fold,seed=seed,regime=reg);timings.append(t)
    result={"scope":"Historical pilot; offline battery monitoring; injected invalid values are not actual hardware faults","environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"machine":platform.machine(),"processor":platform.processor()},"models":18,"panels":len(rows),"checks":sum(r["protected_coordinates"] for r in rows),"max_batch_delta":max(r["batch_delta"] for r in rows),"max_restart_delta":max(r["restart_delta"] for r in rows),"identity_and_integrity_rejections":sum(r["corrupt_or_wrong_identity_rejections"] for r in rows),"guard":{"valid_channel_checks":valid,"false_flags":false_flags,"injected_invalid_values_rejected":faults,"missing_false_flags":missing_flags},"timing_summary":{g:statistics.median([t["median_ratio"] for t in timings if t["regime"]==g]) for g in sorted({t["regime"] for t in timings})},"panels_detail":rows,"timing_samples":timings,"qualification":"HOLD; no actual equipment faults, no board or electrical measurement, no training"}
    pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("panels_detail","timing_samples")},indent=2))
if __name__=="__main__":main()
