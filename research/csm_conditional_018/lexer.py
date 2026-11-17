"""Causal byte lexer with a one-byte delay; no operation or role rules.

The intern table is execution bookkeeping, not neural input. A candidate is
recognized only when its observed right neighbor closes a single-letter token.
No flush at segment end: callers must supply the real following delimiter.
"""
import torch


def init(batch):
    if type(batch) is not int or batch < 1:
        raise ValueError('positive batch required')
    return dict(pending=torch.full((batch,), -1, dtype=torch.long),
                predecessor_is_letter=torch.zeros(batch, dtype=torch.bool),
                table=torch.full((batch, 26), -1, dtype=torch.long),
                nextslot=torch.zeros(batch, dtype=torch.long),
                priorcandidate=torch.full((batch, 47), -1, dtype=torch.long))


def _letter(x):
    return ((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))


def scan(x, state, tied=True):
    """Return resolved, lookahead, 48-slot candidates, newstate, table trace.

    Every input-state tensor is cloned before processing, so any fifth-identity
    rejection leaves the caller's state unchanged, including earlier batch rows.
    """
    if x.ndim != 2 or x.dtype != torch.long or bool(((x < 0) | (x > 255)).any()):
        raise ValueError('raw input must be int64 [B,T] bytes')
    b, tlen = x.shape
    shapes = {'pending':(b,), 'predecessor_is_letter':(b,), 'table':(b,26),
              'nextslot':(b,), 'priorcandidate':(b,47)}
    if set(state) != set(shapes) or any(tuple(state[k].shape) != v for k,v in shapes.items()):
        raise ValueError('incompatible lexer state')
    st = {k:v.to(x.device).clone() for k,v in state.items()}
    resolved=[]; lookahead=[]; candidates=[]; tables=[]
    rows=torch.arange(b,device=x.device)
    for t in range(tlen):
        current=x[:,t]; pending=st['pending']; valid=pending >= 0
        upper=(pending >=65)&(pending <=90)
        identity=valid & upper & ~st['predecessor_is_letter'] & ~_letter(current)
        letter_index=(pending-65).clamp(0,25)
        existing=st['table'][rows,letter_index]
        novel=identity & (existing < 0)
        if bool((novel & (st['nextslot'] >=4)).any()):
            raise ValueError('identity capacity exceeded: fifth distinct identity')
        slot=torch.where(novel,st['nextslot'],existing)
        st['table'][rows[novel],letter_index[novel]]=slot[novel]
        st['nextslot']=st['nextslot']+novel.long()
        r=torch.where(valid,pending,torch.full_like(pending,257))
        if tied:
            r=torch.where(identity,torch.full_like(r,256),r)
        c=torch.where(identity,slot,torch.full_like(slot,-1))
        window=torch.cat((st['priorcandidate'],c[:,None]),1)
        resolved.append(r);lookahead.append(torch.where((current>=65)&(current<=90),torch.full_like(current,256),current))
        candidates.append(window);tables.append(st['table'].clone())
        st['priorcandidate']=window[:,1:]
        st['predecessor_is_letter']=valid & _letter(pending)
        st['pending']=current
    if not tlen:
        return (x.clone(),x.clone(),torch.empty((b,0,48),dtype=torch.long,device=x.device),st,
                torch.empty((b,0,26),dtype=torch.long,device=x.device))
    return torch.stack(resolved,1),torch.stack(lookahead,1),torch.stack(candidates,1),st,torch.stack(tables,1)
