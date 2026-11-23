"""Regressions from independent storage validation, with synthetic data only."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from .session import MemorySession
from .store import MemoryStore, DeletedError, IntegrityError, SCHEMA_VERSION


def claim(store, key, value, source, proposal, revision=0):
    store.propose(key,value,source,proposal_id=proposal,expected_revision=revision)
    store.accept(proposal,authority='synthetic-test')


def checkpoint(store, session='decode', request='one', revision=0, **extra):
    return store.commit_checkpoint(session,request_id=request,
        request_hash=hashlib.sha256(request.encode()).hexdigest(),expected_revision=revision,
        manifest={'model':'test'},payload={'state':request},result={'answer':request},**extra)


@pytest.mark.parametrize('foreign', [('bob','home'),('alice','work')])
def test_foreign_corpus_cannot_change_local_ranking(tmp_path, foreign):
    path=tmp_path/'memory.sqlite3'
    with MemoryStore(path,principal_id='alice',namespace='home') as own:
        own.append_message('a','user','compass',event_id='a')
        own.append_message('a','user','beacon',event_id='b')
        before=own.messages(query='compass beacon')
        with MemoryStore(path,principal_id=foreign[0],namespace=foreign[1]) as other:
            for i in range(50):
                other.append_message('b','user','beacon foreigncanary',event_id=str(i))
        assert own.messages(query='compass beacon') == before
        assert own.messages(query='foreigncanary') == []


def test_unrelated_facts_from_same_source_do_not_evict_direct_match(tmp_path):
    with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice') as store:
        store.append_message('a','user','Orion launches Thursday; ten unrelated stock counts follow.',event_id='source')
        claim(store,'orion.launch','Thursday','source','target')
        for i in range(10):
            claim(store,f'stock.{i}',i,'source',f'p{i}')
        context=MemorySession(store,session_id='new').context('When does Orion launch?')
        assert context['accepted_facts'][0]['key'] == 'orion.launch'
        assert context['accepted_facts'][0]['value'] == 'Thursday'
        assert len(context['accepted_facts']) <= 8


def test_erasure_generation_rejects_inflight_first_checkpoint(tmp_path):
    path=tmp_path/'memory.sqlite3'
    with MemoryStore(path,principal_id='alice') as store:
        store.append_message('a','user','old evidence',event_id='source')
        before=store.erasure_generation()
        with MemoryStore(path,principal_id='bob') as other:
            other.append_message('a','user','foreign evidence',event_id='source')
            other.forget_event('source')
        assert store.erasure_generation() == before
        store.forget_event('source')
        after=store.erasure_generation()
        assert after > before
        with pytest.raises(DeletedError):
            checkpoint(store,expected_erasure_generation=before)
        assert store.load_checkpoint('decode') is None
        store.forget_event('source')
        assert store.erasure_generation() == after
        assert checkpoint(store,expected_erasure_generation=after)['revision'] == 1


def test_empty_session_retirement_changes_erasure_generation(tmp_path):
    with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice') as store:
        before=store.erasure_generation()
        store.forget_session('empty')
        assert store.erasure_generation() > before
        after=store.erasure_generation()
        store.forget_session('empty')
        assert store.erasure_generation() == after


def legacy_database(path):
    """Create the exact v1 table shape from compatible user-visible records.

    Payload/result shapes are unchanged in v2; remove only the newly introduced
    metadata columns/index and label v1. This fixture exercises actual SQL ALTER
    migration and preserves all remaining tables, values and legacy digests.
    """
    with MemoryStore(path,principal_id='alice') as store:
        store.append_message('a','user','Orion launches Thursday',event_id='source')
        claim(store,'orion.launch','Thursday','source','p1')
        store.append_message('b','user','Orion moved to Friday',event_id='correction')
        claim(store,'orion.launch','Friday','correction','p2',1)
        checkpoint(store,request='one')
        checkpoint(store,request='two',revision=1)
    with MemoryStore(path,principal_id='bob',namespace='other') as store:
        store.append_message('b','user','other namespace survives',event_id='source')
        checkpoint(store)
    with sqlite3.connect(path) as db:
        db.execute('DROP INDEX checkpoint_live_outputs')
        db.execute('ALTER TABLE scopes DROP COLUMN erasure_generation')
        db.execute('ALTER TABLE checkpoints DROP COLUMN identity_digest')
        db.execute('ALTER TABLE checkpoint_requests DROP COLUMN identity_digest')
        db.execute('PRAGMA user_version=1')
    # sqlite3's transaction context commits but does not close the connection.
    # A backup fixture must include the checkpointed DB bytes, not an open WAL.
    db.close()


def test_atomic_v1_migration_preserves_all_scopes_records_and_checkpoints(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    legacy_database(path)
    with MemoryStore(path,principal_id='alice') as store:
        assert store.db.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION == 2
        assert store.facts('orion.launch')[0]['value'] == 'Friday'
        assert [f['value'] for f in store.history('orion.launch')] == ['Thursday','Friday']
        assert len(store.messages()) == 2
        assert store.load_checkpoint('decode')['payload'] == {'state':'two'}
        assert store.checkpoint_receipt('decode','one')['result'] == {'answer':'one'}
        assert store.erasure_generation() == 0
        assert checkpoint(store,request='three',revision=2)['revision'] == 3
    with MemoryStore(path,principal_id='bob',namespace='other') as store:
        assert store.messages()[0]['content'] == 'other namespace survives'
        assert store.load_checkpoint('decode')['revision'] == 1


def test_corrupt_legacy_content_rolls_back_entire_migration(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    legacy_database(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE checkpoints SET result='{}' WHERE principal='alice'")
    with pytest.raises(IntegrityError):
        MemoryStore(path,principal_id='alice')
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert 'erasure_generation' not in [r[1] for r in db.execute('PRAGMA table_info(scopes)')]
        assert 'identity_digest' not in [r[1] for r in db.execute('PRAGMA table_info(checkpoints)')]


def test_concurrent_open_migrates_v1_exactly_once(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    legacy_database(path)
    root=Path(__file__).resolve().parents[2]
    code="""import sys
from nanocog.memory.store import MemoryStore
sys.stdin.readline()
with MemoryStore(sys.argv[1],principal_id='alice',timeout=15) as store:
    assert store.facts('orion.launch')[0]['value']=='Friday'
    assert store.load_checkpoint('decode')['revision']==2
"""
    processes=[subprocess.Popen([sys.executable,'-c',code,str(path)],cwd=root,
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
               for _ in range(6)]
    try:
        for process in processes:
            process.stdin.write('\n'); process.stdin.flush()
        for process in processes:
            _stdout,stderr=process.communicate(timeout=25)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill(); process.wait()
    with MemoryStore(path,principal_id='alice') as store:
        assert store.db.execute('PRAGMA user_version').fetchone()[0] == 2
        assert len(store.history('orion.launch')) == 2


def test_legacy_backup_restores_then_migrates_on_open(tmp_path):
    backup=tmp_path/'legacy.sqlite3'
    legacy_database(backup)
    metadata=dict(format='nanocog-memory-backup-v1',schema_version=1,
                  sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),bytes=backup.stat().st_size)
    Path(str(backup)+'.manifest.json').write_text(json.dumps(metadata))
    restored=tmp_path/'restored.sqlite3'
    assert MemoryStore.restore_backup(backup,restored)['schema_version'] == 1
    with MemoryStore(restored,principal_id='alice') as store:
        assert store.facts('orion.launch')[0]['value'] == 'Friday'
        assert store.load_checkpoint('decode')['revision'] == 2


@pytest.mark.parametrize('table,columns',[
    ('checkpoint_requests',('result','digest')),
    ('checkpoints',('payload','digest')),
    ('checkpoints',('result','result_digest')),
])
def test_valid_content_and_checksum_cannot_be_swapped_between_requests(tmp_path,table,columns):
    with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice') as store:
        checkpoint(store,session='first',request='one')
        checkpoint(store,session='second',request='two')
        donor=store.db.execute(f'SELECT * FROM {table} WHERE session=?',('second',)).fetchone()
        store.db.execute(f'UPDATE {table} SET {columns[0]}=?,{columns[1]}=? WHERE session=?',
                         (donor[columns[0]],donor[columns[1]],'first'))
        with pytest.raises(IntegrityError):
            if table == 'checkpoints': store.load_checkpoint('first')
            else: store.checkpoint_receipt('first','one')


@pytest.mark.parametrize('table',['checkpoints','checkpoint_requests'])
def test_malformed_manifest_fails_as_integrity_error(tmp_path,table):
    with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice') as store:
        checkpoint(store)
        store.db.execute(f"UPDATE {table} SET manifest='not JSON'")
        with pytest.raises(IntegrityError):
            if table == 'checkpoints': store.load_checkpoint('decode')
            else: store.checkpoint_receipt('decode','one')


def test_pruning_and_erasure_preserve_receipt_identity_checks(tmp_path):
    with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice') as store:
        store.append_message('a','user','synthetic evidence',event_id='source')
        for i in range(132):
            checkpoint(store,request=str(i),revision=i)
        assert store.checkpoint_receipt('decode','0')['result'] is None
        assert store.checkpoint_receipt('decode','131')['result'] == {'answer':'131'}
        store.forget_event('source')
        for i in range(132):
            assert store.checkpoint_receipt('decode',str(i))['result'] is None
        store.db.execute("UPDATE checkpoint_requests SET revision=999 WHERE request='0'")
        with pytest.raises(IntegrityError): store.checkpoint_receipt('decode','0')


@pytest.mark.parametrize('error_code',[sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED])
def test_journal_startup_retries_only_transient_lock_contention(tmp_path,error_code):
    original_connect=sqlite3.connect
    attempts=[]
    class ContendedConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql == 'PRAGMA journal_mode':
                attempts.append(1)
                if len(attempts)<4:
                    exc=sqlite3.OperationalError('injected transient journal lock')
                    exc.sqlite_errorcode=error_code
                    raise exc
            return super().execute(sql,*args,**kwargs)
    def connect(*args,**kwargs):
        return original_connect(*args,**kwargs,factory=ContendedConnection)
    with patch('nanocog.memory.store.sqlite3.connect',side_effect=connect):
        with MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice',timeout=0.5) as store:
            assert store.db.execute('PRAGMA busy_timeout').fetchone()[0] == 500
            store.append_message('a','user','startup succeeded',event_id='source')
    assert len(attempts) == 4


@pytest.mark.parametrize('error_code,timeout',[(sqlite3.SQLITE_BUSY,0),(sqlite3.SQLITE_IOERR,1)])
def test_journal_startup_does_not_retry_expired_wait_or_io_errors(tmp_path,error_code,timeout):
    original_connect=sqlite3.connect
    attempts=[]
    class FailingConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql == 'PRAGMA journal_mode':
                attempts.append(1)
                exc=sqlite3.OperationalError('injected fatal journal condition')
                exc.sqlite_errorcode=error_code
                raise exc
            return super().execute(sql,*args,**kwargs)
    def connect(*args,**kwargs):
        return original_connect(*args,**kwargs,factory=FailingConnection)
    with patch('nanocog.memory.store.sqlite3.connect',side_effect=connect):
        with pytest.raises(sqlite3.OperationalError):
            MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice',timeout=timeout)
    assert len(attempts) == 1


def test_journal_busy_retry_has_a_finite_total_wait_budget(tmp_path):
    original_connect=sqlite3.connect
    class BusyConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql == 'PRAGMA journal_mode':
                exc=sqlite3.OperationalError('injected persistent journal lock')
                exc.sqlite_errorcode=sqlite3.SQLITE_BUSY
                raise exc
            return super().execute(sql,*args,**kwargs)
    def connect(*args,**kwargs):
        return original_connect(*args,**kwargs,factory=BusyConnection)
    started=time.monotonic()
    with patch('nanocog.memory.store.sqlite3.connect',side_effect=connect):
        with pytest.raises(sqlite3.OperationalError):
            MemoryStore(tmp_path/'memory.sqlite3',principal_id='alice',timeout=0.04)
    # Loose scheduling allowance, but a mistakenly unbounded retry cannot pass.
    assert time.monotonic()-started < 1.0
