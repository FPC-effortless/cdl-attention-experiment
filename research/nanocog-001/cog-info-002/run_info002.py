"""Frozen native feature donor and exact final local G×U transplant actor."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json,os,sys
import torch
from torch.nn import functional as F
from cog_comm.objective import Tokenizer
from cog_comm.role_mixers import build_arm

A='41e1c19a805ac0061252113303d4bf4e5b0e2a3a3c582df4ba67f5763d858761'
CHK={1171:'b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c',
     1371:'517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'}
DONORS=('edit_first','edit_second')
HYBRIDS=('gate','taps','joint')
BR=('base','edit_first_native','edit_second_native',*(f'{b}_{g}' for b in DONORS for g in HYBRIDS))
def need(x,m):
    if not x:raise ValueError(m)
def raw(n,h):
    d=Path(n).read_bytes();need(sha256(d).hexdigest()==h,'input pin '+n);return d
def write(p,x):
    with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype==torch.float32 and torch.equal(
        a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
def native(model,x):
    rec={}
    def observe(_,args,out):
        need(not rec and len(args)==1 and args[0].shape==(*x.shape,32) and out.shape==(*x.shape,4),'L2 site')
        rec['state']=args[0].detach().clone();rec['score']=out.detach().clone()
    h=model.mixers[1].local.offset_gate.register_forward_hook(observe)
    try:z,_=model(x,state=None)
    finally:h.remove()
    need(set(rec)=={'state','score'} and all(bool(torch.isfinite(t).all()) for t in rec.values()),'native capture')
    return z,rec

def mix(local,x,s):
    # Exact original fresh LocalMix layout, mask, evaluation order, and
    # full [B,T,C] pointwise matmul. Only its final result can be injected.
    b,t,_=x.shape
    tv=F.pad(x,(0,0,3,0)).unfold(1,4,1)
    ident=torch.arange(t,device=x.device)[:,None]+torch.arange(4,device=x.device)[None,:]
    w=s.masked_fill(ident<3,-torch.inf).softmax(-1)
    m=(tv*local.depthwise.to(tv)*w.unsqueeze(-2).to(tv)).sum(-1)
    return local.pointwise(m)

def transplant(model,x,rec,donor,mode):
    local=model.mixers[1].local; calls=0
    def replace(_,args,result):
        nonlocal calls
        calls+=1
        delta,cache=result
        need(len(args)==2 and args[1] is None and exact(args[0],rec['state']),'fixed receiver site')
        s=rec['score'];u=rec['state']
        need(exact(mix(local,u,s),delta),'original local recompute not bit exact')
        if mode in ('gate','joint'):
            s=s.clone();s[:,-1]=donor['score'][:,-1]
        if mode in ('taps','joint'):
            u=u.clone();u[:,-4:]=donor['state'][:,-4:]
        mixed=mix(local,u,s)
        new=delta.clone();new[:,-1]=mixed[:,-1]
        need(exact(new[:,:6],delta[:,:6]),'changed earlier local output')
        return new,cache
    h=local.register_forward_hook(replace)
    try:z,_=model(x,state=None)
    finally:h.remove()
    need(calls==1,'unexpected L2 intervention hook calls')
    return z

def main():
    _,cp,sd=sys.argv;seed=int(sd)
    need(not sys.flags.optimize and sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu'
         and seed in CHK and CHK[seed]==cp and os.environ.get('TRAINLAB_SEED')==sd,'exact runtime/channel')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    a=json.loads(raw('actor.json',A))
    need(set(a)=={'schema','prompts','groups'} and a['schema']=='cog-info-002.actor.v1' and
         len(a['prompts'])==len(set(a['prompts']))==768 and len(a['groups'])==32,'exact actor envelope')
    flat=[]
    for g in a['groups']:
        need(set(g)=={'base',*DONORS} and all(len(v)==8 for v in g.values()),'case roster')
        flat.extend(r for v in g.values() for r in v)
        for old,one,two in zip(g['base'],g['edit_first'],g['edit_second'],strict=True):
            x,y,z=(a['prompts'][i].split() for i in (old,one,two))
            edits=[k for k in range(7) if x[k]!=y[k]],[k for k in range(7) if x[k]!=z[k]]
            need(len(x)==len(y)==len(z)==7 and edits==([2],[5]) and y[2]==z[5] and y[2] not in (x[2],x[5]),
                 'fixed one-token render edits')
    need(sorted(flat)==list(range(768)),'no duplicate row')
    ck=torch.load(BytesIO(raw('checkpoint.pt',cp)),map_location='cpu',weights_only=True)
    need(ck['seed']==seed and ck['updates']==1280 and ck['plan']['prefix_ce_weight']==.25 and ck['plan']['clip']==1
         and ck['plan']['arm']=='gated32' and ck['vocab_size']==287,'frozen model')
    t=Tokenizer(tuple(ck['tokenizer_words']))
    ix=[t.encode(p)[1:-1] for p in a['prompts']]
    need(all(len(z)==7 and min(z)>=262 for z in ix),'actor token geometry')
    m=build_arm(287,'gated32',seed).eval().requires_grad_(False)
    m.load_state_dict(ck['state_dict'],strict=True)
    need(m.cfg.width==32 and len(m.mixers)==2 and m.mixers[1].local.window==4,'local edit geometry')
    logits={b:[] for b in BR}; calls=0; earlier=0; selfs=0; repeats=0
    with torch.inference_mode():
        for g in a['groups']:
            batches={k:torch.tensor([ix[r] for r in g[k]],dtype=torch.long) for k in ('base',*DONORS)}
            z,r=native(m,batches['base']);calls+=1
            current={'base':z}
            for d in DONORS:
                donor_logits,n=native(m,batches[d]);calls+=1
                current[d+'_native']=donor_logits
                for tag in HYBRIDS:
                    w=transplant(m,batches['base'],r,n,tag);calls+=1
                    need(exact(w[:,:6],z[:,:6]),'hybrid changed earlier full-model output')
                    earlier+=1;current[d+'_'+tag]=w
            for tag in HYBRIDS:
                w=transplant(m,batches['base'],r,r,tag);calls+=1
                need(exact(w,z),'same-row no-op not bit exact')
                selfs+=1
            restored,_=m(batches['base'],state=None);calls+=1
            need(exact(restored,z),'unhooked replay differs')
            repeats+=1
            for k,w in current.items():
                need(w.shape==(8,7,287) and w.dtype==torch.float32 and bool(torch.isfinite(w).all()),'native outcome geometry')
                logits[k].append(w[:,-1].clone())
    need((calls,earlier,selfs,repeats)==(416,192,96,32) and
         all(exact(v,ck['state_dict'][k]) for k,v in m.state_dict().items()),'scope/frozen weights')
    z=torch.stack([torch.cat(logits[k],dim=0) for k in BR]).contiguous()
    need(z.shape==(9,256,287) and sys.byteorder=='little','wire')
    payload=z.numpy().tobytes(order='C')
    p=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    with (p/'LOGITS.f32le').open('xb') as f:f.write(payload)
    write(p/'intervention_receipt.json',dict(schema='cog-info-002.hybrid.v1',seed=seed,checkpoint_sha256=cp,actor_sha256=A,
         branch_order=BR,shape=list(z.shape),logits_sha256=sha256(payload).hexdigest(),
         tokenizer_words=list(t.words),earlier_hybrid_batches_exact=earlier,same_row_batches_exact=selfs,
         native_replay_batches_exact=repeats,native_mix_recompute_batches_exact=earlier+selfs))
    write(p/'metrics.json',dict(model_forwards=calls,optimizer_updates=0,baseline_rows=256,
         natural_rows=768,intervention_rows=1536,same_row_controls=768,earlier_hybrid_batches_exact=earlier))
if __name__=='__main__':main()
