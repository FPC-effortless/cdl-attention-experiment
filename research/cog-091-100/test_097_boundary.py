"""COG 071 local transaction boundary reused exactly, no real-world writes.

The vendored boundary.py matches the GitHub blob SHA
f4cccf5ac2f7758fd9cb84516a4d39046b5e56ca.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from vendor_097.boundary import Authority, Ledger, Workspace


def test_semantic_proposal_cannot_execute_without_authority():
    with TemporaryDirectory() as folder:
        db=Ledger(Path(folder)/'scratch.db',b'separate-secret');p=db.propose('n1',0,'assign')
        before=db.snapshot()
        assert db.commit(p,None)=='denied'
        assert db.snapshot()==before
        db.db.close()


def test_signature_is_bound_to_exact_target_and_versions():
    with TemporaryDirectory() as folder:
        key=b'only-for-this-fixture';db=Ledger(Path(folder)/'scratch.db',key)
        a=Authority(key);p=db.propose('n1',0,'assign');other=db.propose('n2',1,'assign')
        token=a.issue(p,True,False)
        assert db.commit(other,token)=='denied'
        assert db.commit(p,token)=='committed'
        assert db.commit(p,token)=='duplicate'
        db.db.close()


def test_source_epoch_change_prevents_stale_commit():
    with TemporaryDirectory() as folder:
        key=b'epoch-fixture';db=Ledger(Path(folder)/'scratch.db',key);a=Authority(key)
        p=db.propose('n1',0,'assign');token=a.issue(p,True,False)
        db.db.execute('UPDATE policy SET epoch=epoch+1')
        assert db.commit(p,token)=='stale'
        assert db.db.execute('SELECT COUNT(*) FROM journal').fetchone()[0]==0
        db.db.close()


def test_well_authorized_semantically_wrong_action_can_commit():
    """Invariant: HMAC proves permission for the proposed action, NOT intent."""
    with TemporaryDirectory() as folder:
        key=b'mistaken-intent';db=Ledger(Path(folder)/'scratch.db',key);a=Authority(key)
        # The user really requested 'close', but the model mistakenly chose
        # 'assign'. An authority that permits 'assign' cannot detect this.
        intended='close';p=db.propose('n1',0,'assign')
        assert intended!=p.op
        token=a.issue(p,True,False)
        assert db.commit(p,token)=='committed'
        assert db.db.execute('SELECT value FROM objects WHERE id=0').fetchone()[0]=='assign'
        db.db.close()


def test_provisional_alternatives_do_not_implicitly_authorize():
    w=Workspace();w.stage('hypothesis',(),{'candidate_ops':('close','assign'),'uncertain':True})
    assert w.prepared('hypothesis')['uncertain']
    assert w.discard('hypothesis')==1
    assert not w.nodes
