"""Finite, local next-action semantics; neither network nor actuator calls."""
ROLES=('operator','auditor','requester')
# Local recipient contexts know these fields already, independently of sender.
KNOWN={
 'operator': (('scope','freshness'),('source','scope'),('authority','progress'),('status','proof')),
 'auditor': (('proof','status'),('scope','source'),('freshness','progress'),('authority','status')),
 'requester': (('status','authority'),('scope','source'),('proof','freshness'),('progress','scope')),
}

def receive(role, local, decoded):
    if role not in KNOWN:
        raise ValueError("unknown role")
    # Conflicting present facts fail conservatively instead of overriding evidence.
    if any(k in local and local[k]!=v for k,v in decoded.items()):
        return 'seek evidence'
    f={**local, **decoded}
    def needed(*names):return all(k in f for k in names)
    if not needed('scope','source','freshness'):
        return 'seek evidence'
    if (f['scope'],f['source'],f['freshness'])!=('matched','verified','current'):
        return 'seek evidence'
    if not needed('status','proof'):
        return 'seek evidence'
    if f['status']=='completed' and f['proof']=='accepted':
        return {'operator':'close','auditor':'attest','requester':'notify complete'}[role]
    if role!='operator':
        if f['status']=='blocked':return 'report blocked'
        if needed('progress') and f['progress']=='started':return 'report active'
        return 'seek evidence'
    if not needed('authority','progress'):
        return 'seek evidence'
    if f['authority']=='revoked':return 'seek authority'
    if f['status']=='blocked':return 'report blocked'
    if f['progress']=='started' and f['proof']=='absent':
        return 'resume'
    return 'seek evidence'

def reference(role, complete):
    """Separate complete-prefix interpreter; its rule equivalence is reviewed."""
    if any(complete[k]!=v for k,v in {'scope':'matched','source':'verified','freshness':'current'}.items()):return 'seek evidence'
    if complete['status']=='completed' and complete['proof']=='accepted':
        if role=='operator':return 'close'
        return 'attest' if role=='auditor' else 'notify complete'
    if role=='operator':
        if complete['authority']=='revoked':return 'seek authority'
        if complete['status']=='blocked':return 'report blocked'
        if complete['progress']=='started' and complete['proof']=='absent':return 'resume'
        return 'seek evidence'
    if complete['status']=='blocked':return 'report blocked'
    return 'report active' if complete['progress']=='started' else 'seek evidence'
