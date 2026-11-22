"""Read-only frozen ROLE feature capture; no labels, model edits, or logit consumption."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json,os,sys
import torch
from cog_comm.objective import Tokenizer
from cog_comm.role_mixers import build_arm
from nanocog.model import norm

A='653460a4ead17c87e5b17de9a10b7bb17cdfdedff131a3afca2a182528030b35'
CHECK={1171:'b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c',
       1371:'517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'}
FIELDS=(('literal_raw',32),('literal_normalized',32),('l2_gate_input',32),
        ('l2_gate_probability',4),('pre_output_hidden',32))
def need(x,msg):
    if not x:raise ValueError(msg)
def pin(p,s):
    b=Path(p).read_bytes();need(sha256(b).hexdigest()==s,'input pin '+p);return b
def write(path,val):
    with path.open('x') as f:json.dump(val,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
def capture(model,inputs):
    observed={}
    def add(key,t,w):
        need(key not in observed and t.shape==(*inputs.shape,w) and t.dtype==torch.float32,'read-only boundary '+key)
        observed[key]=t[:,-1].detach().clone()
    def literal(_,args,result):
        need(len(args)==1 and torch.equal(args[0],inputs),'raw input changed')
        add('literal_raw',result,32);add('literal_normalized',norm(result),32)
    def gate(_,args,result):
        need(len(args)==1,'L2 gate args')
        add('l2_gate_input',args[0],32)
        # All four local taps eligible at t=6. Recompute native-dtype
        # softmax on original score tensor shape without modifying inputs.
        add('l2_gate_probability',result.softmax(dim=-1),4)
    def head(_,args):
        need(len(args)==1,'readout boundary args')
        add('pre_output_hidden',args[0],32)
    hooks=(model.embedding.register_forward_hook(literal),
           model.mixers[1].local.offset_gate.register_forward_hook(gate),
           model.output.register_forward_pre_hook(head))
    try:model(inputs,state=None)  # Never access native return or logits.
    finally:
        for hh in hooks:hh.remove()
    need(set(observed)=={k for k,_ in FIELDS},'missing boundary')
    y=torch.cat([observed[k] for k,_ in FIELDS],dim=1); p=observed['l2_gate_probability']
    need(y.shape==(8,132) and bool(torch.isfinite(y).all()) and bool((p>=0).all())
         and bool((p.double().sum(-1)-1.).abs().le(2e-7).all()),'finite normalized traces')
    return y

def main():
    _,native,sd=sys.argv;seed=int(sd)
    need(seed in CHECK and CHECK[seed]==native and sys.version_info[:2]==(3,12)
         and torch.__version__=='2.10.0+cpu' and not sys.flags.optimize
         and os.environ.get('TRAINLAB_SEED')==sd,'exact runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    panel=json.loads(pin('actor.json',A))
    need(set(panel)=={'prompts','schema'} and panel['schema']=='cog-info-001.actor.v1'
         and len(panel['prompts'])==len(set(panel['prompts']))==1536,'actor-only scope')
    ck=torch.load(BytesIO(pin('checkpoint.pt',native)),map_location='cpu',weights_only=True)
    need(ck['seed']==seed and ck['updates']==1280 and ck['plan']['arm']=='gated32'
         and ck['plan']['clip']==1. and ck['plan']['prefix_ce_weight']==.25
         and ck['vocab_size']==287,'frozen checkpoint')
    tok=Tokenizer(tuple(ck['tokenizer_words']))
    ids=[tok.encode(p)[1:-1] for p in panel['prompts']]
    need(all(len(x)==7 and min(x)>=262 for x in ids),'prompt geometry')
    m=build_arm(287,'gated32',seed).eval().requires_grad_(False)
    m.load_state_dict(ck['state_dict'],strict=True)
    need(len(m.mixers)==2 and m.mixers[1].local.window==4 and m.cfg.width==32,'actor geometry')
    seen=[];calls=0
    with torch.inference_mode():
        for i in range(0,len(ids),8):
            seen.append(capture(m,torch.tensor(ids[i:i+8],dtype=torch.long)));calls+=1
    need(all(torch.equal(v,ck['state_dict'][k]) for k,v in m.state_dict().items()),'frozen weights changed')
    t=torch.cat(seen,dim=0)
    need(sys.byteorder=='little' and t.shape==(1536,132) and calls==192,'output dimensions')
    p=Path(os.environ['TRAINLAB_OUTPUT_DIR']); payload=t.contiguous().numpy().tobytes(order='C')
    with (p/'TRACE.f32le').open('xb') as f:f.write(payload)
    write(p/'trace_contract.json',dict(schema='cog-info-001.trace.v1',checkpoint_sha256=native,actor_sha256=A,
      trace_sha256=sha256(payload).hexdigest(),seed=seed,shape=list(t.shape),
      columns=[{'name':k,'width':w} for k,w in FIELDS],native_token_count=7,query_position=6,weights_unchanged=True,
      emitted_response_logits=False,trace_has_truth=False))
    write(p/'metrics.json',dict(trace_rows=len(t),model_forwards=calls,optimizer_updates=0,response_logit_rows=0))
if __name__=='__main__':main()
