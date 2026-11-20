from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from boundary import Authority, Ledger, Workspace

def run():
    key=b'local simulator trust root'
    auth=Authority(key)
    with TemporaryDirectory() as d:
        path=Path(d)/'ledger'; db=Ledger(path,key);before=db.snapshot(); p=db.propose('a',1,'archive')
        token=auth.issue(p,True,True)
        for t in (None,auth.issue(p,True,False),auth.issue(p,False,True),('explicit','invalid')):
            assert db.commit(p,t)=='denied' and db.snapshot()==before
        assert db.commit(replace(p,ticket=2),token)=='denied' and db.snapshot()==before
        try:db.commit(p,token,fail=True)
        except InterruptedError:pass
        else:raise AssertionError('fixture never interrupted')
        db.db.close(); db=Ledger(path,key); assert db.snapshot()==before
        def attempt(_):
            other=Ledger(path,key)
            try:return other.commit(p,token)
            finally:other.db.close()
        with ThreadPoolExecutor(4) as ex: states=list(ex.map(attempt,range(4)))
        assert sorted(states)==['committed']+['duplicate']*3
        stale=replace(p,nonce='new'); snap=db.snapshot()
        assert db.commit(stale,auth.issue(stale,True,True))=='stale' and db.snapshot()==snap
        changed=replace(db.propose('a',1,'reopen'),op='defer')
        assert db.commit(changed,auth.issue(changed,True,True))=='conflict'
        q=db.propose('q',3,'publish'); db.db.execute('UPDATE policy SET epoch=epoch+1'); snap=db.snapshot()
        assert db.commit(q,auth.issue(q,True,True))=='stale' and db.snapshot()==snap
        q=db.propose('b',2,'close'); db.db.execute('UPDATE objects SET version=version+1 WHERE id=4'); snap=db.snapshot()
        assert db.commit(q,auth.issue(q,True,False))=='stale' and db.snapshot()==snap
        w=Workspace()
        for name,parents in [('root',()),('other',()),('a',('root',)),('b',('a','other'))]:
            w.stage(name,parents,q)
        assert w.discard('root')==3 and set(w.nodes)=={'other'} and db.snapshot()==snap
        db.db.close()
    return dict(concurrent=sorted(states),restart_atomic=True,stale_dependency=True,
                epoch_revocation=True,scope_bound=True,rollback_descendants=3,unauthorized_writes=0)
