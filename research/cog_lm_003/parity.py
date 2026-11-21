"""Verify sparse predictions and gradients against the native 002 token-level path."""
import copy
from common import torch, Readout
from fit import sparse


def run():
    torch.set_num_threads(1)
    torch.manual_seed(33)
    x=torch.randn(1,311,64)
    ids=torch.randint(-10,75,(1,311))
    emb=torch.randn(75,64)
    select=torch.tensor([15,16,127,128,230,310])
    for short in (False,True):
        head=Readout(64,True)
        ref=copy.deepcopy(head)
        pred,_=sparse(head,emb,x[0],ids[0],select,short)
        p=[]
        width=128 if short else 311
        for h,idx in zip(x.split(width,1),ids.split(width,1)):
            p.append(ref(h,idx,emb)[0])
        logits=torch.cat(p,1)[0,select]
        e=float((pred-logits).abs().max().detach())
        torch.testing.assert_close(pred,logits,atol=1.2e-5,rtol=1e-5)
        pred[:,4].mean().backward()
        logits[:,4].mean().backward()
        g=max((p.grad-q.grad).abs().max().item() for p,q in zip(head.parameters(),ref.parameters()))
        assert g<2e-5,g
        print('short',short,'max output discrepancy',e,'max gradient discrepancy',g)
if __name__=='__main__':run()
