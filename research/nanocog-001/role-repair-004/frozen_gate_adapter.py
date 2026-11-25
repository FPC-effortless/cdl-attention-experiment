"""Diagnostic wrapper, exact seven-token fixture only.

It routes the literal embedding before norm/smear to the adapter. The
original model receives the unchanged token sequence and has no registered
updates. No optimizer or native-model invocation on import.
"""
import torch
from torch import nn
from torch.nn import functional as F

class FrozenGateAdapter(nn.Module):
    def __init__(self,backbone,adapter):
        super().__init__()
        if len(backbone.mixers)!=2 or backbone.cfg.width!=32:
            raise ValueError('expected exact two-layer width-32 ROLE backbone')
        if backbone.mixers[1].local.window != 4:
            raise ValueError('incorrect native lag geometry')
        self.backbone=backbone.requires_grad_(False).eval()
        self.adapter=adapter

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, tokens, *, enabled=True, capture_features=False):
        if tokens.ndim!=2 or tokens.shape[1]!=7 or tokens.dtype!=torch.long:
            raise ValueError('exact seven-token fresh prompt only')
        if any(p.requires_grad for p in self.backbone.parameters()) or self.backbone.training:
            raise ValueError('backbone freeze broken')
        if type(enabled) is not bool or type(capture_features) is not bool:
            raise ValueError('invalid forward mode')
        if not enabled and not capture_features:
            return self.backbone(tokens,state=None)[0]
        capture={}
        # The original model computes e = norm(self.embedding(tokens)).
        # This hook returns None: no write into the frozen embedding path.
        def raw_embed(_, arguments, result):
            if capture.get('literal') is not None or not torch.equal(arguments[0],tokens):
                raise ValueError('literal embedding call changed')
            capture['literal']=result.detach()
        def final_gate(_,args,scores):
            x=args[0]
            if capture.get('called') or capture.get('literal') is None or                 scores.shape!=(*tokens.shape,4) or x.shape!=capture['literal'].shape:
                raise ValueError('unexpected L2 offset-gate call')
            capture['called']=True
            # Fresh LocalMix: no state, left-padding three zero inputs. Unfold
            # yields B,T,C,4; lag order is oldest history through current.
            taps=F.pad(x.detach(),(0,0,3,0)).unfold(1,4,1).transpose(-2,-1)
            t=torch.arange(7,device=x.device)
            eligible=(t[:,None]+torch.arange(4,device=x.device)[None,:])>=3
            active=t==6
            if capture_features:
                with torch.no_grad():
                    capture['features']=self.adapter.features(capture['literal'][:,-1:],taps[:,-1:]).detach()
            if enabled:
                return self.adapter.apply_to(scores,capture['literal'],taps,eligible,active=active)
        hooks=(self.backbone.embedding.register_forward_hook(raw_embed),
               self.backbone.mixers[1].local.offset_gate.register_forward_hook(final_gate))
        try:
            z,_=self.backbone(tokens,state=None)
        finally:
            for h in hooks:h.remove()
        if not capture.get('called'):
            raise ValueError('L2 offset gate not invoked')
        return (z,capture['features']) if capture_features else z
