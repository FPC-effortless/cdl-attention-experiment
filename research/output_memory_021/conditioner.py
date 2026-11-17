"""Frozen LANGUAGE_BASELINE_001 conditioner. Literary training is not task competence."""
import importlib.util
import sys
from pathlib import Path
import hashlib
import torch


def load_conditioner(checkpoint,source,device='cpu'):
    checkpoint,source=Path(checkpoint),Path(source)
    # Only known experiment schema; weights_only blocks arbitrary checkpoint pickle execution.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['arm']!='csm' or saved['seed'] not in (11,23,37):raise ValueError('unexpected retained model')
    name='_native_csm021_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    state=saved['state_dict'];v,d=state['embedding.weight'].shape;s=state['first_address.weight'].shape[0]
    o=state['operator_embedding'].shape[0];inner=state['transform.0.weight'].shape[0]
    model=module.RecordRewrite(module.Config(v,width=d,slots=s,operators=o,inner=inner))
    model.load_state_dict(state,strict=True);model.to(device).eval();model.requires_grad_(False)
    return model,{'seed':saved['seed'],'updates':saved['updates'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'condition_dim':s*d+s*s+s+2*d}


def flatten_state(state):
    # Position omitted consistently: scalar batched position includes padding length.
    return torch.cat([t.flatten(1) for t in (state.values,state.links,state.cursor,state.history)],1).detach().clone()


@torch.no_grad()
def prefix_condition(model,prefixes):
    """Only caller-supplied prefix bytes, no labels, no true answer lengths.
    Right padded lanes preserve all tensor state fields; features do not include position.
    """
    if not prefixes:raise ValueError('nonempty batch')
    device=next(model.parameters()).device;state=model.init_state(len(prefixes),device)
    for p in range(max(map(len,prefixes),default=0)):
        active=torch.tensor([p<len(x) for x in prefixes],device=device)
        tokens=torch.tensor([x[p] if p<len(x) else 0 for x in prefixes],device=device)
        _,new=model.step(tokens,state)
        for field in ('values','links','cursor','history'):
            old_value=getattr(state,field);new_value=getattr(new,field)
            shape=(len(prefixes),)+(1,)*(new_value.ndim-1)
            setattr(new,field,torch.where(active.reshape(shape),new_value,old_value))
        state=new
    return flatten_state(state)


class FrozenSession:
    """Volatile learned hidden state only; caller owns accepted-text sequencing.
    Does not implement authoritative facts or a verified semantic interpreter.
    """
    def __init__(self,model):
        self.model=model
        self.state=model.init_state(1,next(model.parameters()).device)
        self.ingested_bytes=0

    @torch.no_grad()
    def consume(self,text):
        if not isinstance(text,bytes):raise TypeError('bytes required')
        for token in text:
            _,self.state=self.model.step(torch.tensor([token],device=self.state.values.device),self.state)
        self.ingested_bytes+=len(text)

    def feature(self):
        return flatten_state(self.state)
