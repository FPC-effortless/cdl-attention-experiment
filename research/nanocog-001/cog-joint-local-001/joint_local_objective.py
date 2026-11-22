"""The locked per-step training loss. Import is inert, no model or data access."""
ARMS={'gated32_mean':('gated32',False),'gated32_tail':('gated32',True),'attention32_tail':('attention32',True)}
SEEDS=(2371,2471,2571)
ENDPOINTS=(128,1280)

def need(ok,m):
    if not ok:raise ValueError(m)

def objective(z,x,y,foils,*,tail):
    import torch
    from torch.nn import functional as F
    from math import log
    need(z.shape==(8,7,287) and x.shape==(8,7) and y.shape==(8,) and
         z.dtype==torch.float32 and x.dtype==y.dtype==torch.long and type(tail)is bool,
         'exact loss input shapes')
    f=z[:,6,:]
    row=F.cross_entropy(f,y,reduction='none')
    plain=row.mean()
    prefix=F.cross_entropy(z[:,:6,:].reshape(-1,287),x[:,1:].reshape(-1))
    if foils is not None:
        need(foils.shape==(4,) and foils.dtype==torch.long and bool((y[2:6]!=foils).all()),
             'four distinct gold/foil rows')
        mix=row[:6].mean()
        soft=torch.logsumexp(row[:6],dim=0)-log(6.)
        weighted=(6*(.5*mix+.5*soft)+row[6]+row[7])/8 if tail else plain
        logits=f[2:6]
        margins=logits.gather(1,y[2:6,None]).flatten()-logits.gather(1,foils[:,None]).flatten()
        edit=F.softplus(1.-margins).mean()
    else:
        weighted=plain
        mix=soft=edit=z.new_zeros(())
    return weighted+.25*prefix+.25*edit,dict(
        plain_answer_ce=plain,answer_aggregation=weighted,mixed_mean=mix,mixed_logmeanexp=soft,
        prefix_ce=prefix,edit_margin_softplus=edit)
