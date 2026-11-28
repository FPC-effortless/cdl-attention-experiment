"""Scalar forgetting of the incoming delta-memory state before its error-correcting write.
State s[t] = f[t] * s[t-1] + k[t] outer (g[t] * (v[t] - k[t]^T f[t] s[t-1])).
Exact causal triangular solve in chunks, no full-history attention, constant carried state.
"""
import math, torch
from model import delta

def retained_delta(k,v,q,g,logf,s,chunk=32):
    out=[]
    for kk,vv,qq,gg,ll in zip(k.split(chunk,1),v.split(chunk,1),q.split(chunk,1),g.split(chunk,1),logf.split(chunk,1)):
        cum=ll.cumsum(1); decay=(cum[:,:,None]-cum[:,None,:]).clamp_max(0).exp()
        f=cum.exp(); gg=gg[...,None]
        system=(gg*(kk@kk.transpose(-1,-2))*decay).tril(-1)
        rhs=gg*(vv-f[...,None]*(kk@s))
        u=torch.linalg.solve_triangular(system,rhs,upper=False,unitriangular=True)
        out.append(f[...,None]*(qq@s)+((qq@kk.transpose(-1,-2))*decay).tril()@u)
        s=f[:,-1,None,None]*s+kk.transpose(-1,-2)@(decay[:,-1,:,None]*u)
    return torch.cat(out,1),s

def reference(k,v,q,g,logf,s):
    out=[]
    for i in range(k.size(1)):
        ki=k[:,i];s=logf[:,i,None,None].exp()*s
        u=g[:,i,None]*(v[:,i]-torch.einsum('bi,bij->bj',ki,s))
        s=s+ki[:,:,None]*u[:,None,:];out.append(torch.einsum('bi,bij->bj',q[:,i],s))
    return torch.stack(out,1),s

def learnable_log_retention(g,raw,scale=.02):
    """No-write HOLD exactly: g=0 means f=1; positive g enables bounded decay.
    The raw scalar may come from an unused projection row. Nonzero bias is explicit.
    """
    return -scale*g*raw.sigmoid()
