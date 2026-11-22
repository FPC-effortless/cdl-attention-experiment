"""Raw-embedding and final native mixture adapter; optional and unlaunched."""
import torch
from torch import nn

class FrozenValueRepairPort(nn.Module):
    """Only an existing L2 local mixture reaches the added residual.

    Call-site experiment owns the exact frozen reference checkpoint.
    This source never constructs a model or loads any checkpoint.
    """
    def __init__(self,backbone,residual):
        super().__init__()
        if getattr(backbone.cfg,'width',None)!=32 or len(backbone.mixers)!=2 or backbone.mixers[1].local.window!=4:
            raise ValueError('expected exact native local geometry')
        self.backbone=backbone.requires_grad_(False).eval()
        self.value_residual=residual
    def train(self,mode=True):
        super().train(mode);self.backbone.eval()
        return self
    def forward(self,tokens,*,enabled=False):
        if type(enabled) is not bool:
            raise ValueError('enabled must be explicit bool')
        if not enabled:
            return self.backbone(tokens,state=None)
        if tokens.ndim!=2 or tokens.shape[1]!=7 or tokens.dtype!=torch.long or                 self.backbone.training or any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError('expected frozen evaluation backbone and fresh seven-token input')
        b=tokens.shape[0]; native=self.backbone.mixers[1].local; record={}
        def literal(_,args,embed):
            if len(args)!=1 or not torch.equal(args[0],tokens) or record or                     embed.shape!=(b,7,32) or embed.dtype!=torch.float32:
                raise ValueError('unexpected raw embedding callback')
            record['literal']=embed[:,-1].detach()
        def receipt(_,args,score):
            if set(record)!={'literal'} or len(args)!=1 or args[0].shape!=(b,7,32) or                     args[0].dtype!=torch.float32 or score.shape!=(b,7,4) or score.dtype!=torch.float32:
                raise ValueError('unexpected L2 callback')
            # Fresh state, last token: no padded positions among four taps.
            record['taps']=args[0][:,-4:,:].detach()
            record['gate']=score.softmax(-1)[:,-1].detach()
        def mixture(_,args):
            if set(record)!={'literal','taps','gate'} or len(args)!=1 or                     args[0].shape!=(b,7,32) or args[0].dtype!=torch.float32:
                raise ValueError('unexpected native pointwise boundary')
            mixed=args[0]
            if not bool(torch.isfinite(mixed).all()):raise ValueError('nonfinite native mixture')
            last=self.value_residual.apply_to(mixed[:,-1],record['literal'],record['taps'],record['gate'],enabled=True)
            edited=mixed.clone();edited[:,-1]=last
            record['patched']=True
            return (edited,)
        hooks=(self.backbone.embedding.register_forward_hook(literal),
               native.offset_gate.register_forward_hook(receipt),
               native.pointwise.register_forward_pre_hook(mixture))
        try:out=self.backbone(tokens,state=None)
        finally:
            for h in hooks:h.remove()
        if not record.get('patched',False):raise ValueError('missing mixer boundary')
        return out
