"""A local transaction boundary and per-session speculative dependency graph."""
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import sqlite3

OPERATIONS = ('close','reopen','assign','defer','prioritize','label','archive','publish')
HIGH = {'archive','publish'}
LOW = {'label','defer'}

@dataclass(frozen=True)
class Proposal:
    nonce: str
    ticket: int
    op: str
    version: int
    dependency: int
    dependency_version: int
    epoch: int

    def payload(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(',',':'))

class Authority:
    def __init__(self,key):
        self.key=key

    def issue(self,p,permitted,explicit):
        if not permitted or (p.op in HIGH and not explicit):
            return None
        level='explicit' if explicit else 'ordinary'
        return level,hmac.digest(self.key,(level+p.payload()).encode(),'sha256').hex()

class Ledger:
    def __init__(self,path,key):
        self.db=sqlite3.connect(path,isolation_level=None)
        self.key=key
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS objects(id INTEGER PRIMARY KEY,version INTEGER,value TEXT);
          CREATE TABLE IF NOT EXISTS journal(id TEXT PRIMARY KEY,payload TEXT);
          CREATE TABLE IF NOT EXISTS policy(id INTEGER PRIMARY KEY,epoch INTEGER);
          INSERT OR IGNORE INTO policy VALUES(1,0);
        """)
        self.db.executemany("INSERT OR IGNORE INTO objects VALUES(?,0,'open')",[(i,) for i in range(5)])

    def version(self,i):
        return self.db.execute('SELECT version FROM objects WHERE id=?',(i,)).fetchone()[0]

    def epoch(self):
        return self.db.execute('SELECT epoch FROM policy').fetchone()[0]

    def propose(self,nonce,ticket,op):
        return Proposal(nonce,ticket,op,self.version(ticket),4,self.version(4),self.epoch())

    def commit(self,p,token,fail=False):
        if not token or p.op not in OPERATIONS or p.ticket not in range(4) or p.dependency!=4:
            return 'denied'
        level,sig=token
        if level not in ('ordinary','explicit') or (p.op in HIGH and level!='explicit'):
            return 'denied'
        if not hmac.compare_digest(sig,hmac.digest(self.key,(level+p.payload()).encode(),'sha256').hex()):
            return 'denied'
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior=self.db.execute('SELECT payload FROM journal WHERE id=?',(p.nonce,)).fetchone()
            if prior:
                status='duplicate' if prior[0]==p.payload() else 'conflict'
            elif self.epoch()!=p.epoch or self.version(p.ticket)!=p.version or self.version(4)!=p.dependency_version:
                status='stale'
            else:
                self.db.execute('UPDATE objects SET value=?,version=version+1 WHERE id=?',(p.op,p.ticket))
                if fail:
                    raise InterruptedError('fixture interruption before dependent write')
                self.db.execute("UPDATE objects SET value=?,version=version+1 WHERE id=4",(p.nonce,))
                self.db.execute('INSERT INTO journal VALUES(?,?)',(p.nonce,p.payload()))
                status='committed'
            self.db.execute('COMMIT')
            return status
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def snapshot(self):
        return [self.db.execute('SELECT * FROM '+name+' ORDER BY 1').fetchall() for name in ('objects','journal','policy')]

class Workspace:
    def __init__(self):
        self.nodes={}
        self.undo=[]

    def stage(self,key,parents,payload):
        if key in self.nodes or any(p not in self.nodes for p in parents):
            raise ValueError('invalid stage dependencies')
        self.nodes[key]=(tuple(parents),payload)

    def discard(self,key):
        removed={key}
        for node,(parents,_) in list(self.nodes.items()):
            if node in removed or removed.intersection(parents):
                removed.add(node)
                self.undo.append((node,self.nodes.pop(node)))
        return len(removed)

    def prepared(self,key):
        return self.nodes[key][1]
