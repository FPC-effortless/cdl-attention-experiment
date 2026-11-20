"""State writes, versioned authority and exact idempotency under one transaction."""
import json
import sqlite3


class World:
    def __init__(self,path=":memory:"):
        self.db=sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE authority (id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);
        INSERT INTO authority VALUES(1,0);
        CREATE TABLE state (k TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE journal (id TEXT PRIMARY KEY, epoch INTEGER, patch TEXT NOT NULL);
        """)

    def epoch(self):
        return self.db.execute("SELECT epoch FROM authority WHERE id=1").fetchone()[0]

    def advance(self):
        with self.db:
            self.db.execute("UPDATE authority SET epoch=epoch+1 WHERE id=1")

    def commit(self,key,epoch,patch):
        self.db.execute("BEGIN IMMEDIATE")
        with self.db:
            if epoch!=self.epoch():
                return "stale"
            normalized=json.dumps(patch,sort_keys=True)
            old=self.db.execute("SELECT patch FROM journal WHERE id=?",(key,)).fetchone()
            if old:
                return "duplicate" if old[0]==normalized else "conflict"
            self.db.execute("INSERT INTO journal VALUES(?,?,?)",(key,epoch,normalized))
            for field,value in patch.items():
                if value is None:
                    self.db.execute("DELETE FROM state WHERE k=?",(field,))
                else:
                    self.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(k) DO UPDATE SET value=excluded.value",
                                    (field,json.dumps(value)))
            return "committed"

    def state(self):
        return {k:json.loads(v) for k,v in self.db.execute("SELECT k,value FROM state")}

