"""Forward-only native ROLE gate probe. No truth labels or optimizer."""
import json,os,sys
from hashlib import sha256
from pathlib import Path
import torch
from cog_comm.role_mixers import build_arm
from cog_comm.objective import Tokenizer
from query_gate_probe import probe

def req(ok,msg):
    if not ok:raise ValueError(msg)
def pinned(p,sha):
    b=Path(p).read_bytes();req(sha256(b).hexdigest()==sha,'pinned input mismatch');return b
def main():
    _,ap,cp,seed=sys.argv
    seed=int(seed)
    req(sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu','wrong runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    raw=json.loads(pinned('actor.json',ap))
    req(set(raw)=={'prompts','schema'} and raw['schema']=='cog-query-probe.actor.v1' and len(set(raw['prompts']))==len(raw['prompts'])==256,'bad actor')
    from io import BytesIO
    ck=torch.load(BytesIO(pinned('checkpoint.pt',cp)),map_location='cpu',weights_only=True)
    req(ck['seed']==seed and ck['updates']==1280 and ck['vocab_size']==287 and ck['plan']['clip']==1.0 and ck['plan']['prefix_ce_weight']==.25,'bad checkpoint')
    vocab=Tokenizer(tuple(ck['tokenizer_words']))
    model=build_arm(vocab.size,'gated32',seed).eval(); model.requires_grad_(False)
    model.load_state_dict(ck['state_dict'],strict=True)
    x0={k:v.clone() for k,v in model.state_dict().items()}
    rows=[vocab.encode(p)[1:-1] for p in raw['prompts']]
    req(all(len(s)==7 and min(s)>=262 for s in rows),'tokenization changed')
    l={k:[] for k in ('native','opposite','sham')}; d={}
    for i in range(0,len(rows),8):
        r=probe(model,torch.tensor(rows[i:i+8],dtype=torch.long))
        for k,v in r['logits'].items():l[k].extend(v.tolist())
        for k,v in r['diagnostics'].items():d.setdefault(k,[]).extend(v.tolist())
    req(all(torch.equal(t,model.state_dict()[k]) for k,t in x0.items()),'frozen weights mutated')
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (out/'predictions.json').write_text(json.dumps(dict(arm='gated32',seed=seed,updates=1280,
           checkpoint_sha256=cp,actor_sha256=ap,tokenizer_words=list(vocab.words),logits=l,diagnostics=d),
           sort_keys=True,allow_nan=False,separators=(',',':')))
    (out/'metrics.json').write_text(json.dumps(dict(prediction_rows=256,model_forwards=160,optimizer_updates=0)))
if __name__=='__main__':main()
