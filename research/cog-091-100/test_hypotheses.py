from hypotheses import HypothesisNet,targets,score
from semantic import make_data,to_tensor
import torch

def test_two_hot_targets_and_forward():
    data=make_data(9107);cache={};sample=data['train'][:12]
    t=targets(sample)
    for i,r in enumerate(sample):
        assert int(t[i].sum())==(2 if r['answer']==6 else 1)
    x,l,_=to_tensor(sample,cache)
    a,h=HypothesisNet()(x,l)
    assert a.shape==(12,7) and h.shape==(12,6)

def test_no_self_attention_or_authentication_bypass():
    m=HypothesisNet()
    assert not any(isinstance(z,torch.nn.MultiheadAttention) for z in m.modules())
    assert not hasattr(m,'commit')
