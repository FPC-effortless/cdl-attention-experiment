"""Reference adapter around unchanged archived CSM; eager correctness, not speed."""
import importlib.util
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
with zipfile.ZipFile(ROOT/'artifacts/csm-original-scope-reconstructed-20261109/CSM-GRAPH-001-verified-project.zip') as z:
    source=z.read('CSM-GRAPH-001/csm-foundation/experiments/csm_foundation_001/model.py')
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/'base.py';p.write_bytes(source)
    spec=importlib.util.spec_from_file_location('_csm_archived_base',p)
    base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

@dataclass
class State:
    core: object
    regulation: torch.Tensor

class Regulated(nn.Module):
    def __init__(self, config, variant='multi', traces=12):
        super().__init__()
        if traces<=0 or traces%3: raise ValueError('positive multiple of three required')
        self.core=base.RecordRewrite(config)
        self.driver=nn.Linear(config.width,traces)
        self.injection=nn.Linear(traces,config.width,bias=False)
        if variant=='instant': lam=torch.zeros(traces,dtype=torch.float64)
        elif variant=='single': lam=torch.full((traces,),-1/32,dtype=torch.float64).exp()
        elif variant=='multi': lam=(-1/torch.tensor([4.,32.,256.],dtype=torch.float64)).exp().repeat_interleave(traces//3)
        else: raise ValueError(variant)
        self.register_buffer('decay',lam)
    def init_state(self,batch,device):
        return State(self.core.init_state(batch,device),self.driver.weight.new_zeros(batch,self.driver.out_features))
    def step(self,tokens,state,valid=True,reset=False,return_trace=False):
        # Flags apply to the whole batch; asynchronous document boundaries need
        # independently scheduled streams. No silent partial-batch reset.
        if type(valid) is not bool or type(reset) is not bool: raise TypeError('scalar bool flags required')
        if reset and not valid: raise ValueError('reset on padding')
        if not valid: return None,state,{'padding':True}
        if reset: state=self.init_state(tokens.shape[0],tokens.device)
        trace=[]
        def inject(module,args,output):
            front=args[0][:,:self.core.config.width]
            signal=torch.tanh(self.driver(front))
            r=self.decay*state.regulation+(1-self.decay)*signal
            trace.append(r)
            return output+self.injection(r)
        # Scoped hook keeps archived forward and its surrogate derivatives intact.
        handle=self.core.controller.register_forward_hook(inject)
        try: logits,core,details=self.core.step(tokens,state.core,return_trace=True)
        finally: handle.remove()
        return logits,State(core,trace[0]),details
    def forward(self,tokens):
        state=self.init_state(tokens.shape[0],tokens.device);ys=[]
        for t in range(tokens.shape[1]):
            y,state,_=self.step(tokens[:,t],state);ys.append(y)
        return torch.stack(ys,1)
