"""Parameter-neutral candidate: context-conditioned decay before delta update.
Only address row zero (unused in HH) in each layer is repurposed. Typed slots and
recurrent scan remain unchanged. At initialization learned mode equals fixed
write mode mathematically, not the unmodified parent. No extra carried state.
"""
import math, inspect, textwrap, types
import torch
from model import CogLayer, NativeLM, delta
import model as parent
from retention import retained_delta,learnable_log_retention
MODES=('native','time512','write128','learned_write','learned_scalar')
MAX_DECAY=.02
INIT_BIAS=math.log((math.log(2)/128/MAX_DECAY)/(1-math.log(2)/128/MAX_DECAY))

def update(k,v,q,g,s,mode,raw):
    if mode=='native':return delta(k,v,q,g,s)
    if mode=='time512':ll=torch.full_like(g,-math.log(2)/512)
    elif mode=='write128':ll=-math.log(2)/128*g
    elif mode in ('learned_write','learned_scalar'):ll=learnable_log_retention(g,raw,MAX_DECAY)
    else:raise ValueError(mode)
    return retained_delta(k,v,q,g,ll,s)

# Source-checked, one localized change; avoids a duplicated HH cell definition.
_src=textwrap.dedent(inspect.getsource(CogLayer.forward))
_needle='mb,nb=delta(k,vbg,q,gb,bg)'
assert _src.count(_needle)==1
_key='k=F.normalize(self.address(u)[...,8:],dim=-1)'
assert _src.count(_key)==1
_src=_src.replace(_key,'all_addr=self.address(u);k=F.normalize(all_addr[...,8:],dim=-1)')
_src=_src.replace(_needle,'mb,nb=update(k,vbg,q,gb,bg,self._retention_mode,(self.address.bias[0].expand_as(gb) if self._retention_mode=="learned_scalar" else all_addr[...,0]))')
_ns=dict(vars(parent));_ns['update']=update;exec(_src,_ns);_forward=_ns['forward']

def adapt(weights,mode):
    if mode not in MODES:raise ValueError(mode)
    m=NativeLM('HH');m.load_state_dict(weights,strict=True)
    for block in m.core.blocks:
        block._retention_mode=mode
        if mode!='native':block.forward=types.MethodType(_forward,block)
        if mode in ('learned_write','learned_scalar'):
            with torch.no_grad():block.address.weight[0].zero_();block.address.bias[0].fill_(INIT_BIAS)
    return m
