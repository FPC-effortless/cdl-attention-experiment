"""Inert numeric fixture: no model construction, data access, or optimizer."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys
import torch
from torch.nn.functional import cross_entropy, softplus

def load(name, path):
    spec=spec_from_file_location(name,path)
    mod=module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

torch.set_num_threads(1)
here=Path(__file__).resolve().parent
old=load("role001_original_objective",here.parent/"cog-role-001/smoke/cog_comm/objective.py")
new=load("role002_weighted_objective",here/"cog_comm/objective.py")
z=torch.sin(torch.arange(8*7*287,dtype=torch.float32)).reshape(8,7,287).requires_grad_()
y=torch.tensor([[262,263],[264,265],[266,267],[268,269]])
pfx=torch.arange(8*7).reshape(8,7)+100
pfx[:,-1]=-1
length=torch.full((8,),7)
ref, rlast=old.paired_loss(z,length,y,lm_targets=pfx)
unit, ulast=new.paired_loss(z,length,y,lm_targets=pfx,prefix_ce_weight=1.)
got, glast=new.paired_loss(z,length,y,lm_targets=pfx,prefix_ce_weight=.25)
gold=.25*cross_entropy(z.reshape(-1,287),pfx.flatten(),ignore_index=-1)+cross_entropy(z[:,-1],y.flatten())+softplus(
    z[torch.arange(8),-1,y.flip(1).flatten()]-z[torch.arange(8),-1,y.flatten()]).mean()
gref,gunit,ggot,ggold=(torch.autograd.grad(value,z,retain_graph=True)[0] for value in (ref,unit,got,gold))
assert torch.equal(ref,unit) and torch.equal(gref,gunit)
assert torch.equal(rlast,ulast) and torch.equal(rlast,glast)
torch.testing.assert_close(got,gold,atol=1e-7,rtol=1e-7)
torch.testing.assert_close(ggot,ggold,atol=1e-7,rtol=1e-7)
assert torch.equal(ggot[:,-1],gref[:,-1]) and torch.equal(ggot[:,:-1],.25*gref[:,:-1])
print("Inert fixture PASS: weight=1 exact compatibility, .25 formula/gradient, same answer gradient")
