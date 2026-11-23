"""Durability failures use temporary files and separate processes, never real memory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from .store import MemoryStore, ConflictError, IntegrityError, DeletedError, _patched_wal


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=self.root/'memory.sqlite3'

    def tearDown(self):
        self.temp.cleanup()

    def open(self, **kw):
        return MemoryStore(self.path,principal_id=kw.pop('principal_id','alice'),**kw)

    def worker(self, code):
        return subprocess.run([sys.executable,'-c',code,str(self.path)],capture_output=True,text=True,
                              timeout=30,env=dict(os.environ,PYTHONPATH=str(Path(__file__).parents[2])))

    def source(self,s,event='source',text='Amara owns the watch',session='chat-a'):
        return s.append_message(session,'user',text,event_id=event)

    def proposal(self,s,proposal='p1',value='Amara',revision=0,source='source'):
        return s.propose('watch.owner',value,source,proposal_id=proposal,expected_revision=revision)

    def test_fresh_process_recall_scope_and_duplicate_conflict(self):
        result=self.worker("from nanocog.memory.store import MemoryStore; import sys,os\n"
            "s=MemoryStore(sys.argv[1],principal_id='alice'); "
            "s.append_message('chat-a','user','The spare key is in the blue drawer',event_id='source'); os._exit(0)")
        self.assertEqual(result.returncode,0,result.stderr)
        with self.open() as s:
            self.assertIn('blue drawer',s.recall('spare key')['observations'][0]['content'])
            self.assertEqual(len(s.messages()),1)
            with self.assertRaises(ConflictError):self.source(s,text='different bytes')
        with self.open(principal_id='bob') as s:self.assertEqual(s.recall('spare key')['observations'],[])
        with self.open(namespace='elsewhere') as s:self.assertEqual(s.messages(),[])

    def test_proposals_do_not_appear_as_facts_and_corrections_keep_history(self):
        with self.open() as s:
            self.source(s); self.proposal(s)
            self.assertEqual(s.facts(),[])
            first=s.accept('p1',authority='user-confirmed')
            self.assertEqual(first['revision'],1)
            self.source(s,event='correction',text='Tobi now owns the watch')
            self.proposal(s,'p2','Tobi',1,'correction')
            second=s.accept('p2',authority='user-confirmed')
            self.assertEqual(second['revision'],2)
            self.assertEqual([f['value'] for f in s.history('watch.owner')],['Amara','Tobi'])
            self.assertEqual(s.accept('p1',authority='user-confirmed')['revision'],1)
            self.assertEqual(s.facts()[0]['value'],'Tobi')

    def test_accepted_values_have_indexed_recall_and_old_value_is_removed(self):
        with self.open() as s:
            self.source(s,text='Confirmed update from the trusted host')
            self.proposal(s,value='violetnotebook')
            s.accept('p1',authority='test')
            self.assertEqual(s.recall('violetnotebook')['facts'][0]['value'],'violetnotebook')
            self.proposal(s,proposal='p2',value='ambernotebook',revision=1)
            s.accept('p2',authority='test')
            self.assertEqual(s.recall('violetnotebook')['facts'],[])
            self.assertEqual(s.recall('ambernotebook')['facts'][0]['value'],'ambernotebook')
            s.forget_event('source')
            self.assertEqual(s.recall('ambernotebook')['facts'],[])
        self.assertNotIn(b'violetnotebook',self.path.read_bytes().lower())
        self.assertNotIn(b'ambernotebook',self.path.read_bytes().lower())

    def test_checkpoint_output_bytes_are_bounded_but_deduplication_remains(self):
        with patch('nanocog.memory.store.CHECKPOINT_OUTPUT_BYTES',100):
            with self.open() as s:
                for n in range(3):
                    s.commit_checkpoint('decode',request_id=str(n),request_hash=str(n)*64,
                        expected_revision=n,manifest={'model':'x'},payload={'n':n},result={'text':'x'*60})
                self.assertIsNone(s.checkpoint_receipt('decode','0')['result'])
                self.assertIsNone(s.checkpoint_receipt('decode','1')['result'])
                self.assertIsNotNone(s.checkpoint_receipt('decode','2')['result'])
                self.assertEqual(s.load_checkpoint('decode')['revision'],3)

    def test_concurrent_stale_fact_updates_have_one_winner(self):
        with self.open() as s:
            self.source(s)
            self.proposal(s,'p-a','Amara'); self.proposal(s,'p-b','Tobi')
        barrier=threading.Barrier(2)
        def attempt(p):
            with self.open() as s:
                barrier.wait(timeout=10)
                try:s.accept(p,authority='test'); return 'committed'
                except ConflictError:return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(attempt,['p-a','p-b']))
        self.assertEqual(sorted(results),['committed','conflict'])
        with self.open() as s:self.assertEqual(len(s.history('watch.owner')),1)

    def test_process_dies_halfway_through_fact_commit(self):
        with self.open() as s:self.source(s); self.proposal(s)
        code="""from nanocog.memory.store import MemoryStore
import sys, os
s=MemoryStore(sys.argv[1],principal_id='alice')
def trace(sql):
    if sql.startswith('INSERT INTO fact_history'): os._exit(23)
s.db.set_trace_callback(trace)
s.accept('p1',authority='test')
"""
        result=self.worker(code); self.assertEqual(result.returncode,23,result.stderr)
        with self.open() as s:
            self.assertEqual(s.facts(),[])
            self.assertEqual(s.history('watch.owner'),[])
            self.assertEqual(s.accept('p1',authority='test')['revision'],1)

    def test_process_dies_after_commit_before_ack_then_exact_retry(self):
        with self.open() as s:self.source(s); self.proposal(s)
        result=self.worker("from nanocog.memory.store import MemoryStore; import sys,os\n"
                           "s=MemoryStore(sys.argv[1],principal_id='alice'); "
                           "s.accept('p1',authority='test'); os._exit(24)")
        self.assertEqual(result.returncode,24,result.stderr)
        with self.open() as s:
            self.assertEqual(s.accept('p1',authority='test')['revision'],1)
            self.assertEqual(len(s.history('watch.owner')),1)

    def test_erasure_removes_dependencies_facts_search_and_retries(self):
        with self.open() as s:
            self.source(s,text='Private violet notebook')
            s.append_message('chat-a','assistant','violet notebook answer',event_id='answer',source_event_ids=['source'])
            s.append_message('chat-b','assistant','copied violet notebook',event_id='copy',source_event_ids=['answer'])
            self.proposal(s,value='violet notebook'); s.accept('p1',authority='test')
            result=s.forget_event('source')
            self.assertEqual(result['derived_messages_erased'],2)
            self.assertEqual(s.messages(query='violet'),[])
            self.assertEqual(s.facts(),[])
            self.assertTrue(all(f['value'] is None for f in s.history('watch.owner')))
            with self.assertRaises(DeletedError):self.source(s,text='Private violet notebook')
            with self.assertRaises(DeletedError):s.accept('p1',authority='test')
            with self.assertRaises(DeletedError):s.get_message('answer')
            with self.assertRaises(DeletedError):s.append_message('new','assistant','late reply',event_id='late',source_event_ids=['source'])
            self.assertTrue(s.forget_event('source')['already_erased'])
            s.backup(self.root/'after-erasure.db')
        restored=self.root/'restored.db'
        MemoryStore.restore_backup(self.root/'after-erasure.db',restored)
        with MemoryStore(restored,principal_id='alice') as s:
            self.assertEqual(s.messages(),[])
            with self.assertRaises(DeletedError):s.get_message('source')

    def test_session_forget_is_idempotent_and_new_id_required(self):
        with self.open() as s:
            self.source(s)
            s.forget_session('chat-a')
            self.assertTrue(s.forget_session('chat-a')['already_erased'])
            with self.assertRaises(DeletedError):s.append_message('chat-a','user','new text',event_id='new')
            s.append_message('chat-b','user','new text',event_id='new')

    def test_erased_unique_word_is_removed_from_current_search_index_bytes(self):
        unique='erasureprobexvqznolifetimecopy'
        with self.open() as s:
            self.source(s,text=unique)
            self.assertTrue(s.messages(query=unique))
            s.forget_event('source')
        self.assertNotIn(unique.encode(),self.path.read_bytes().lower())

    def test_backup_restore_export_and_corruption(self):
        backup=self.root/'backup.db'
        with self.open() as s:
            self.source(s); self.proposal(s); s.accept('p1',authority='test')
            s.backup(backup)
            export=s.export_json(self.root/'export.json')
            payload=Path(export['path']).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),export['sha256'])
            self.assertEqual(json.loads(payload)['tables']['facts'][0]['key'],'watch.owner')
        restore=self.root/'restored.db'
        MemoryStore.restore_backup(backup,restore)
        with MemoryStore(restore,principal_id='alice') as s:self.assertEqual(s.facts()[0]['value'],'Amara')
        with self.assertRaises(FileExistsError):MemoryStore.restore_backup(backup,restore)
        with backup.open('r+b') as f:f.seek(200); f.write(b'changed')
        with self.assertRaises(IntegrityError):MemoryStore.restore_backup(backup,self.root/'bad.db')
        self.assertFalse((self.root/'bad.db').exists())

    def test_stale_sqlite_sidecars_cannot_override_restored_backup(self):
        backup=self.root/'backup.db'
        with self.open() as s:self.source(s); s.backup(backup)
        for suffix in ('-wal','-shm','-journal'):
            dest=self.root/('restore'+suffix+'.db')
            sidecar=Path(str(dest)+suffix); sidecar.write_bytes(b'stale private data')
            with self.assertRaises(FileExistsError):MemoryStore.restore_backup(backup,dest)
            with self.assertRaises(FileExistsError):MemoryStore(dest,principal_id='alice')
            self.assertFalse(dest.exists())

    def test_dangling_symlink_destinations_not_followed(self):
        backup=self.root/'backup.db'
        with self.open() as s:self.source(s); s.backup(backup)
        dest=self.root/'dangling.db'; dest.symlink_to(self.root/'nonexistent.db')
        with self.assertRaises(FileExistsError):MemoryStore.restore_backup(backup,dest)
        self.assertFalse((self.root/'nonexistent.db').exists())

    def test_unknown_schema_rejected_without_starting_empty(self):
        with self.open() as s:self.source(s); s.db.execute('PRAGMA user_version=999')
        with self.assertRaises(IntegrityError):self.open()

    def test_corrupt_message_and_fact_not_silently_returned(self):
        with self.open() as s:
            self.source(s); self.proposal(s); s.accept('p1',authority='test')
            s.db.execute("UPDATE facts SET value='\"Tobi\"'")
            with self.assertRaises(IntegrityError):s.facts()
            s.db.execute("UPDATE messages SET content='edited'")
            with self.assertRaises(IntegrityError):s.messages()

    def test_untrusted_search_syntax_and_bounded_retrieval(self):
        with self.open() as s:
            for i in range(20):self.source(s,event=str(i),text='violet '+('x'*1000))
            result=s.recall('violet OR NEAR() "*"',max_chars=500)
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=False,separators=(',',':'))),500)
            self.assertTrue(result['truncated'])
            self.assertEqual(s.messages(query='***'),[])

    def test_finite_json_only(self):
        with self.open() as s:
            self.source(s)
            with self.assertRaises(ValueError):self.proposal(s,value=float('nan'))
            with self.assertRaises(ValueError):self.proposal(s,value=object())

    def test_safe_journal_fallback_on_old_sqlite(self):
        for version in ((3,51,3),(3,53,1),(3,50,7),(3,44,6)):
            self.assertTrue(_patched_wal(version))
        for version in ((3,51,2),(3,50,6),(3,44,5),(3,45,1)):
            self.assertFalse(_patched_wal(version))
        with patch('nanocog.memory.store.sqlite3.sqlite_version_info',(3,45,1)):
            with self.open() as s:
                self.assertEqual(s.journal_mode,'delete')
                self.source(s)


if __name__=='__main__':unittest.main()
