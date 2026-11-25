"""Label-free full-vocabulary replay for exact native and fitted adapter states."""
from hashlib import sha256
from io import BytesIO
import json,os,sys
from pathlib import Path
import torch
from cog_comm.objective import Tokenizer
from fit_adapter import BASE,SEEDS,ARMS,base_model,need,state_hash
from tap_gate_residual import TapGateResidual
from frozen_gate_adapter import FrozenGateAdapter

PANEL='6915a754aa598d5608f0041850b3897ecfeef0bd8731835815e24333ba6c98d1'
def pinned(path,d):
    raw=Path(path).read_bytes()
    need(sha256(raw).hexdigest()==d,'wrong actor or adapter bytes')
    return raw
def main():
    arm,seed,updates,weight_sha=sys.argv[1:]
    seed,updates=int(seed),int(updates)
    need(sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu' and not sys.flags.optimize,'wrong runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    actor=json.loads(pinned('actor.json',PANEL))
    need(set(actor)=={'prompts','schema'} and actor['schema']=='cog-comm-role-001.actor.v1' and
         len(actor['prompts'])==len(set(actor['prompts']))==960,'wrong held synthetic projection')
    base,f=base_model(Path('checkpoint.pt'));initial=state_hash(base); tok=Tokenizer(tuple(f['tokenizer_words']))
    model=base
    if arm=='native':
        need(seed==1371 and updates==0 and weight_sha=='none','wrong native control')
    else:
        need(arm in ARMS and seed in SEEDS and updates in (128,1280),'wrong adapter request')
        a=torch.load(BytesIO(pinned('adapter.pt',weight_sha)),map_location='cpu',weights_only=True)
        need(a['schema']=='cog-role-repair-004.adapter.v1' and a['format']==1 and a['arm']==arm and
             a['updates']==updates and a['seed']==seed and a['backbone_sha256']==BASE and
             a['backbone_state_sha256']==initial and a['tokenizer_words']==list(tok.words) and
             a['vocab_size']==tok.size==287,'wrong adapter metadata')
        adapt=TapGateResidual(arm=arm);adapt.load_state_dict(a['state_dict'],strict=True)
        need(a['adapter_state_sha256']==state_hash(adapt) and all(torch.isfinite(p).all() for p in adapt.parameters()),'adapter tensors changed')
        model=FrozenGateAdapter(base,adapt).eval().requires_grad_(False)
    x=[tok.encode(z)[1:-1] for z in actor['prompts']]
    need(all(len(z)==7 and min(z)>=262 for z in x),'unexpected token geometry')
    logits=[]
    with torch.inference_mode():
        for i in range(0,len(x),8):
            t=torch.tensor(x[i:i+8],dtype=torch.long)
            v=model(t,state=None)[0] if arm=='native' else model(t)
            v=v[:,-1]
            need(v.shape==(8,287) and bool(torch.isfinite(v).all()),'wrong logit shape')
            logits.extend(v.tolist())
    need(state_hash(base)==initial,'changed frozen base')
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (out/'predictions.json').write_text(json.dumps(dict(arm=arm,seed=seed,updates=updates,backbone_sha256=BASE,
      adapter_sha256=weight_sha,actor_sha256=PANEL,tokenizer_words=list(tok.words),vocab_size=287,logits=logits),
      sort_keys=True,allow_nan=False,separators=(',',':')))
    (out/'metrics.json').write_text(json.dumps(dict(prediction_rows=960,model_forwards=120,optimizer_updates=0)))
if __name__=='__main__':main()
