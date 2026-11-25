"""Native geometry-only actor: exact old weights, capture one natural gate forward/batch."""
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json, math, os, sys
import torch
from cog_comm.role_mixers import build_arm
from cog_comm.objective import Tokenizer
from probability_dose import match_probabilities
from dose_summary import summarize

def need(ok,msg):
    if not ok:raise ValueError(msg)
def pinned(name,pin):
    b=Path(name).read_bytes()
    need(sha256(b).hexdigest()==pin,'staged input mismatch '+name)
    return b
def dump(out,name,d):
    with (out/name).open('x') as f:json.dump(d,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
def normalize(p):
    s=math.fsum(p)
    need(abs(s-1.0)<=2e-7 and all(math.isfinite(x) and 0<=x<=1 for x in p),'invalid FP32 probabilities')
    return tuple(x/s for x in p)
def main():
    ap,cp,seed=sys.argv[1:]; seed=int(seed)
    need(seed in (1171,1371) and sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu','runtime/seed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    actor=json.loads(pinned('actor.json',ap))
    need(set(actor)=={'prompts','schema'} and actor['schema']=='cog-query-probe.actor.v1' and
         len(set(actor['prompts']))==len(actor['prompts'])==256,'wrong actor scope')
    ck=torch.load(BytesIO(pinned('checkpoint.pt',cp)),map_location='cpu',weights_only=True)
    need(ck['seed']==seed and ck['updates']==1280 and ck['vocab_size']==287 and ck['plan']['arm']=='gated32'
         and ck['plan']['clip']==1.0 and ck['plan']['prefix_ce_weight']==.25,'wrong checkpoint')
    model=build_arm(287,'gated32',seed).eval();model.requires_grad_(False)
    model.load_state_dict(ck['state_dict'],strict=True)
    need(len(model.mixers)==2 and model.mixers[1].local.window==4,'wrong layer')
    tok=Tokenizer(tuple(ck['tokenizer_words']))
    ids=[tok.encode(p)[1:-1] for p in actor['prompts']]
    need(all(len(x)==7 and min(x)>=262 for x in ids),'bad tokens')
    need(all(ids[i][:6]==ids[i+1][:6] and ids[i][-1]!=ids[i+1][-1] for i in range(0,256,2)),'donors must differ solely in query')
    records=[]; doses=[]; observed_tensors=[]; calls=0
    def capture(_, _args, score):
        need(score.shape==(8,7,4) and score.dtype==torch.float32 and bool(torch.isfinite(score).all()),'wrong gate score')
        # All four taps are eligible at the final token; do softmax before
        # slicing to preserve LocalMix's original tensor rank/geometry.
        observed_tensors.append((score[:,-1].detach().clone(),score.softmax(-1)[:,-1].detach().clone()))
    hook=model.mixers[1].local.offset_gate.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for s in range(0,256,8):
                need(not observed_tensors,'stale capture')
                model(torch.tensor(ids[s:s+8],dtype=torch.long),state=None)
                calls+=1
                need(len(observed_tensors)==1,'unexpected layer access')
                st,prob=observed_tensors.pop()
                scores,ps=st.tolist(),prob.tolist()
                for i,(score,p) in enumerate(zip(scores,ps,strict=True)):
                    d=match_probabilities(normalize(p),normalize(ps[i^1]))
                    records.append(dict(row=s+i,donor=(s+i)^1,scores_fp32=score,p_fp32=p,
                                        intended=asdict(d)))
                    doses.append(d)
    finally:hook.remove()
    need(calls==32 and len(records)==256 and all(torch.equal(v,ck['state_dict'][k])
         for k,v in model.state_dict().items()),'native scope/weight integrity')
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    dump(out,'geometry.json',dict(schema='cog-role-gate-002.geometry.v1',seed=seed,actor_sha256=ap,checkpoint_sha256=cp,
          rows=records,summary=summarize(doses),intended_radius_at_least_point1=sum(d.radius>=.1 for d in doses)))
    dump(out,'metrics.json',dict(probability_rows=len(records),model_forwards=calls,optimizer_updates=0,intervention_forwards=0))
if __name__=='__main__':main()
