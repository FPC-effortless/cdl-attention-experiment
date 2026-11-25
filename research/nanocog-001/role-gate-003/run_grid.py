"""Native label-free ROLE-GATE-003; only L2 final gate scores are replaced."""
from hashlib import sha256
from io import BytesIO
import json,math,os,sys
from pathlib import Path
from array import array
import torch
from cog_comm.role_mixers import build_arm
from cog_comm.objective import Tokenizer
from query_gate_probe import capture,replace_query_gate,paired_query_mask
from simplex_grid import FLOOR,choices

def need(ok,msg):
    if not ok:raise ValueError(msg)
def pinned(name,pin):
    b=Path(name).read_bytes();need(sha256(b).hexdigest()==pin,'input pin '+name)
    return b
def dump(out,name,d):
    with (out/name).open('x') as f:json.dump(d,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
def realizable(x):
    """Nearly log(ideal); deterministic floor-coordinate round-up, only if required."""
    z=torch.tensor(x,dtype=torch.float64)
    s=z.log().float()
    edge=(z==FLOOR)
    p=None
    for n in range(5):
        # Same FP32 softmax input stride/full outer shape as the 8×7×4 native gate.
        point=s.expand(8,7,4).contiguous().softmax(-1)[:,-1]
        need(torch.equal(point,point[:1].expand_as(point)),'batch-nonuniform static point')
        p=point[0].double()
        if bool((p>=FLOOR).all()):break
        s=torch.where(edge,torch.nextafter(s,torch.full_like(s,torch.inf)),s)
    need(bool((p>=FLOOR).all()) and abs(math.fsum(p.tolist())-1)<2e-7
         and float((p-z).abs().max())<=2e-7,'bad realized fixed probability')
    return s,p.tolist(),n

def run_fixed_point(model,x,replacement,realized):
    hits=0
    expected=torch.tensor(realized,dtype=torch.float32)
    def rewrite(_,args,scores):
        nonlocal hits
        hits+=1
        need(scores.shape==(8,7,4) and scores.dtype==torch.float32,'gate geometry changed')
        update=scores.clone()
        update[:,-1]=replacement
        # Last token has all lags valid on a fresh 7-token prompt. Using
        # the complete rank-3 tensor matches actual per-row kernel geometry.
        rt=update.softmax(-1)[:,-1]
        need(torch.equal(rt,expected.expand_as(rt)) and bool((rt.double()>=FLOOR).all()),
             'runtime softmax probability differs or crosses floor')
        return update
    h=model.mixers[1].local.offset_gate.register_forward_hook(rewrite)
    try:
        with torch.inference_mode():out,_=model(x,state=None)
    finally:h.remove()
    need(hits==1,'fixed-point hook did not run exactly once')
    return out

def main():
    ap,cp,seed=sys.argv[1:];seed=int(seed)
    need(seed in (1171,1371) and sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu','runtime/seed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    actor=json.loads(pinned('actor.json',ap))
    need(set(actor)=={'prompts','schema'} and actor['schema']=='cog-query-probe.actor.v1' and
         len(set(actor['prompts']))==len(actor['prompts'])==256,'actor')
    ck=torch.load(BytesIO(pinned('checkpoint.pt',cp)),map_location='cpu',weights_only=True)
    need(ck['seed']==seed and ck['updates']==1280 and ck['vocab_size']==287 and ck['plan']['arm']=='gated32'
         and ck['plan']['clip']==1.0 and ck['plan']['prefix_ce_weight']==.25,'checkpoint')
    model=build_arm(287,'gated32',seed).eval();model.requires_grad_(False)
    model.load_state_dict(ck['state_dict'],strict=True)
    tok=Tokenizer(tuple(ck['tokenizer_words']))
    ids=[tok.encode(s)[1:-1] for s in actor['prompts']]
    need(all(len(x)==7 and min(x)>=262 for x in ids),'encoding')
    grid=choices()
    fixed={n:realizable(p) for n,p in grid.items()}
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    branches=('native',*grid)
    wire={k:array('f') for k in branches}
    total=0
    for i in range(0,len(ids),8):
        x=torch.tensor(ids[i:i+8],dtype=torch.long)
        paired_query_mask(x)  # verifies pair labels never involved
        same=torch.arange(8,dtype=torch.long)
        native, scores=capture(model,x);total+=1
        ctrl=replace_query_gate(model,x,scores,(1,),same);total+=1
        need(torch.equal(ctrl,native),'same-row rewrite changes logits')
        inline={'native':native}
        for name,(logscore,realized,_) in fixed.items():
            z=run_fixed_point(model,x,logscore,realized);total+=1
            need(torch.equal(z[:,:6],native[:,:6]),'changed earlier logits')
            inline[name]=z
        with torch.inference_mode(): repeat,_=model(x,state=None)
        total+=1
        need(torch.equal(repeat,native),'native replay differs after operations')
        for name,z in inline.items():
            need(z.shape==(8,7,287) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'bad logits')
            wire[name].extend(z[:,-1].flatten().tolist())
    need(total==448 and all(len(a)==256*287 and a.itemsize==4 for a in wire.values()),'wrong executed scope')
    need(all(torch.equal(v,ck['state_dict'][k]) for k,v in model.state_dict().items()),'checkpoint mutated')
    bp=out/'logits.f32le'
    with bp.open('xb') as f:
        for n in branches:
            a=wire[n]
            if sys.byteorder!='little':a.byteswap()
            a.tofile(f)
    dump(out,'grid_receipt.json',dict(schema='cog-role-gate-003.grid.v1',seed=seed,actor_sha256=ap,checkpoint_sha256=cp,
         vocab_size=287,tokenizer_words=list(tok.words),shape=[12,256,287],branch_order=branches,
         logits_sha256=sha256(bp.read_bytes()).hexdigest(),intended_probabilities=grid,
         realized_probabilities={n:p for n,(_,p,_) in fixed.items()},
         floor_ulp_adjustments={n:k for n,(_,_,k) in fixed.items()},native_parity_batches=32,
         same_row_parity_batches=32,early_logits_parity_grid_batches=352))
    dump(out,'metrics.json',dict(probability_choices=11,prediction_rows=256,model_forwards=total,optimizer_updates=0))
if __name__=='__main__':main()
