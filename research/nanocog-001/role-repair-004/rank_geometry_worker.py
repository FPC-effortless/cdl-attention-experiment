"""Forward-only rank/causal-parity fixture; never imports supervision or optimizer."""
from io import BytesIO
import json,os,sys
from hashlib import sha256
from pathlib import Path
import torch
from cog_comm.objective import Tokenizer
from cog_comm.role_mixers import build_arm
from frozen_gate_adapter import FrozenGateAdapter
from tap_gate_residual import TapGateResidual

PANEL='7776cd8e328a32043253ffcc463a60fb97037ceb694eb23d7df0632ce2aaac4b'
ARM=('tap_compare','literal_control')
def req(ok,msg):
    if not ok:raise ValueError(msg)
def bytes_at(path,pin):
    b=Path(path).read_bytes();req(sha256(b).hexdigest()==pin,'exact input changed')
    return b
def main():
    _,cp,seed=sys.argv
    seed=int(seed)
    req(sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu','unexpected runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=json.loads(bytes_at('actor.json',PANEL))
    req(set(source)=={'schema','prompts'} and source['schema']=='cog-comm-role-001.actor.v1' and
        len(source['prompts'])==len(set(source['prompts']))==2560,'wrong input projection')
    ck=torch.load(BytesIO(bytes_at('checkpoint.pt',cp)),map_location='cpu',weights_only=True)
    req(ck['seed']==seed and ck['updates']==1280 and ck['plan']['prefix_ce_weight']==.25 and ck['plan']['clip']==1
        and ck['vocab_size']==287,'wrong checkpoint')
    vocab=Tokenizer(tuple(ck['tokenizer_words']))
    ids=[vocab.encode(p)[1:-1] for p in source['prompts']]
    req(all(len(x)==7 and min(x)>=262 for x in ids),'unexpected ID geometry')
    model=build_arm(287,'gated32',seed).eval().requires_grad_(False)
    model.load_state_dict(ck['state_dict'],strict=True)
    native={n:v.detach().clone() for n,v in model.state_dict().items()}
    arms={}
    for name in ARM:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261126)
            m=TapGateResidual(arm=name)
        req(sum(p.numel() for p in m.parameters())==552,'matched parameter accounting')
        arms[name]=FrozenGateAdapter(model,m).eval()
    req(all(torch.equal(a,b) for a,b in zip(arms[ARM[0]].adapter.parameters(),arms[ARM[1]].adapter.parameters())),'mismatched initialization')
    rec={k:[] for k in ARM};changed={k:0 for k in ARM};forward=0
    with torch.inference_mode():
        for i in range(0,len(ids),8):
            batch=torch.tensor(ids[i:i+8],dtype=torch.long)
            unpatched=None
            if i<64:
                unpatched=model(batch,state=None)[0];forward+=1
                req(bool(torch.isfinite(unpatched).all()),'nonfinite native logits')
            for k,m in arms.items():
                z,f=m(batch,enabled=True,capture_features=True);forward+=1
                req(f.shape==(8,1,4,68) and bool(torch.isfinite(f).all()),'wrong feature data')
                rec[k].append(f[:,0])
                if unpatched is not None:
                    req(torch.equal(z,unpatched),'zero residual must be exact')
                    m.adapter.last.weight.copy_(torch.linspace(-.25,.25,8).reshape(1,-1))
                    modified=m(batch);forward+=1
                    m.adapter.last.weight.zero_()
                    req(bool(torch.isfinite(modified).all()) and torch.equal(modified[:,:6],unpatched[:,:6]),
                        'nonzero probe changed earlier outputs')
                    changed[k]+=int((modified[:,-1]!=unpatched[:,-1]).any(-1).sum())
        req(all(torch.equal(v,model.state_dict()[n]) for n,v in native.items()),'frozen backbone changed')
    for k in ARM:req(changed[k]>0,'inert probe cannot influence any final score')
    raw=torch.stack([torch.cat(rec[k],dim=0) for k in ARM]).contiguous()
    req(sys.byteorder=='little' and raw.shape==(2,2560,4,68) and raw.dtype==torch.float32,'wire geometry')
    payload=raw.numpy().tobytes(order='C')
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (out/'FEATURES.f32le').write_bytes(payload)
    (out/'geometry.json').write_text(json.dumps(
      dict(schema='cog-role-repair-004.witness.v1',seed=seed,actor_sha256=PANEL,checkpoint_sha256=cp,
           arms=list(ARM),shape=list(raw.shape),features_sha256=sha256(payload).hexdigest(),probe_changed_rows=changed,
           zero_parity_rows_per_arm=64,earlier_parity_rows_per_arm=64),
      sort_keys=True,allow_nan=False)+'\n')
    (out/'metrics.json').write_text(json.dumps(dict(model_forwards=forward,optimizer_updates=0,feature_rows_per_arm=10240))+ '\n')
if __name__=='__main__':main()
