"""Frozen native terminal-logit observer; no labels, probes, or model updates."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json,os,sys
import torch
from cog_comm.objective import Tokenizer
from cog_comm.role_mixers import build_arm

A='653460a4ead17c87e5b17de9a10b7bb17cdfdedff131a3afca2a182528030b35'
CHECK={1171:'b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c',
       1371:'517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'}
def need(x,msg):
    if not x:raise ValueError(msg)
def pin(p,s):
    b=Path(p).read_bytes();need(sha256(b).hexdigest()==s,'input pin '+p);return b
def write(path,val):
    with path.open('x') as f:json.dump(val,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
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
    seen=[];calls=0; hidden=[]
    def head_input(_,args):
        need(len(args)==1 and args[0].shape==(8,7,32),'exact pre-readout boundary')
        hidden.append(args[0][:,-1].detach().clone())
    hook=m.output.register_forward_pre_hook(head_input)
    with torch.inference_mode():
        for i in range(0,len(ids),8):
            z,_=m(torch.tensor(ids[i:i+8],dtype=torch.long),state=None)
            v=z[:,-1]
            need(v.shape==(8,287) and bool(torch.isfinite(v).all()), 'output shape/finite')
            seen.extend(v.tolist());calls+=1
    hook.remove()
    need(len(hidden)==192 and sys.byteorder=='little','incomplete pre-output verification')
    head_sha=sha256(torch.cat(hidden).contiguous().numpy().tobytes(order='C')).hexdigest()
    need(all(torch.equal(v,ck['state_dict'][k]) for k,v in m.state_dict().items()),'frozen weights changed')
    need(len(seen)==1536 and calls==192,'actor exact budget')
    p=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    write(p/'predictions.json',dict(schema='cog-info-001.native-logits.v1',seed=seed,checkpoint_sha256=native,
      actor_sha256=A,pre_output_sha256=head_sha,vocab_size=287,tokenizer_words=list(tok.words),logits=seen))
    write(p/'metrics.json',dict(prediction_rows=len(seen),model_forwards=calls,optimizer_updates=0))
if __name__=='__main__':main()
