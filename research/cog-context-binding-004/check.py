"""Control-equivalence, stream parity, and prefix causality checks."""
import copy
from experiment import ContextBlock, build_model, ids, scan, torch

def verify():
    torch.set_num_threads(1)
    native=build_model("C",9731)
    x=torch.randint(256,(3,95))
    for mode in ("none","write","read","both"):
        variant=copy.deepcopy(native)
        for b in variant.parent.core.blocks:
            b.__class__=ContextBlock
            b.context_write=mode in ("write","both")
            b.context_read=mode in ("read","both")
        with torch.no_grad():
            for training in (True,False):
                native.train(training)
                variant.train(training)
                z=variant(x)
                left,st=variant(x[:,:31],carry=True)
                right=variant(x[:,31:],state=st)
                assert (torch.cat([left,right],1)-z).abs().max()<2e-4
                if mode=="none":
                    assert (native(x)-z).abs().max()<2e-5
        # At the English digit all four names have the same local suffix.
        x_names=ids([f"The log lists {n} at 2" for n in ("Ada","Ben","Cia","Dan")])
        with torch.no_grad():
            h,_,aux=variant.parent.local(x_names,variant.parent.core.emb)
            key=aux[1][:,-1]
            assert torch.equal(key,key[:1].expand_as(key))
            if mode in ("write","both"):
                b=variant.parent.core.blocks[0]
                value,gate=b.inp(b.norm(h)).chunk(2,-1)
                gate=(.5+.49*gate.sigmoid()).unflatten(-1,(-1,2)).mean(-1,keepdim=True).expand_as(value.unflatten(-1,(-1,2))).flatten(-2)
                recurrent,_=scan(value.tanh(),gate)
                key=key+b.address(recurrent[:,-1])[...,:8]
                assert float((key-key[:1]).abs().max())>1e-5
        print(mode,"passed")
if __name__=="__main__":
    verify()
