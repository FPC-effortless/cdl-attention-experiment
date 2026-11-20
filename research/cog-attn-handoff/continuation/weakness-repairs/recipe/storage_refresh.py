"""Process-level contention and bitemporal compaction on a local WAL store."""
import multiprocessing as mp
import os
import random
import resource
import sqlite3
import tempfile
import time
from pathlib import Path

CTX=mp.get_context("fork")


def race_worker(file,identity,atomic,read_barrier,advance_barrier,queue):
    db=sqlite3.connect(file,timeout=20,isolation_level=None)
    selection,epoch=db.execute("SELECT selector,epoch FROM authority").fetchone()
    read_barrier.wait(15)
    advance_barrier.wait(15)
    start=time.perf_counter_ns()
    rejects=0
    for index in range(100):
        if index:
            selection,epoch=db.execute("SELECT selector,epoch FROM authority").fetchone()
        command=f"{identity}:{index}"
        db.execute("BEGIN IMMEDIATE")
        actual=db.execute("SELECT selector,epoch FROM authority").fetchone()
        if atomic and actual!=(selection,epoch):
            rejects+=1
            selection,epoch=actual
            db.execute("ROLLBACK")
            db.execute("BEGIN IMMEDIATE")
            selection,epoch=db.execute("SELECT selector,epoch FROM authority").fetchone()
        claimed=selection+1
        inserted=db.execute("INSERT OR IGNORE INTO journal VALUES(?,?,?,?)",
                            (command,claimed,selection,epoch)).rowcount
        # Same delivery repeated inside a transaction, then again via restart replay in audit.
        assert not db.execute("INSERT OR IGNORE INTO journal VALUES(?,?,?,?)",
                              (command,claimed,selection,epoch)).rowcount
        if inserted:
            db.execute("UPDATE counters SET value=value+?",(claimed,))
        db.execute("COMMIT")
    queue.put((identity,rejects,time.perf_counter_ns()-start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def interrupted_worker(file):
    db=sqlite3.connect(file,isolation_level=None)
    db.execute("BEGIN IMMEDIATE")
    db.execute("UPDATE counters SET value=99999999")
    os._exit(12)


def race(file, mode):
    with sqlite3.connect(file) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE authority(selector INTEGER, epoch INTEGER);
        INSERT INTO authority VALUES(0,0);
        CREATE TABLE counters(value INTEGER);
        INSERT INTO counters VALUES(0);
        CREATE TABLE journal(id TEXT PRIMARY KEY, delta INTEGER, selector INTEGER, epoch INTEGER);
        """)
    read_barrier,advance_barrier=CTX.Barrier(5),CTX.Barrier(5)
    queue=CTX.Queue()
    workers=[CTX.Process(target=race_worker,args=(file,i,mode=="atomic",read_barrier,advance_barrier,queue))
             for i in range(4)]
    for process in workers: process.start()
    read_barrier.wait(15)
    with sqlite3.connect(file) as db:
        db.execute("UPDATE authority SET selector=1,epoch=1")
    advance_barrier.wait(15)
    workers_stats=[queue.get(timeout=25) for worker in workers]
    for process in workers:
        process.join(10)
        assert process.exitcode==0
    # Hard exit with an open transaction. A following connection verifies recovery.
    crash=CTX.Process(target=interrupted_worker,args=(file,))
    crash.start()
    crash.join(10)
    assert crash.exitcode==12
    with sqlite3.connect(file) as db:
        journal=db.execute("SELECT id,delta,selector,epoch FROM journal ORDER BY id").fetchall()
        before=db.execute("SELECT value FROM counters").fetchone()[0]
        assert db.execute("INSERT OR IGNORE INTO journal VALUES(?,?,?,?)",journal[0]).rowcount==0
        after=db.execute("SELECT value FROM counters").fetchone()[0]
    return dict(mode=mode,journal=journal,value=after,pre_replay=before,worker_stats=sorted(workers_stats),
                peak_parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


QUERY="""SELECT v FROM events WHERE k=? AND valid<=? AND tx<=?
         ORDER BY valid DESC,tx DESC LIMIT 1"""


def value(connection,params):
    row=connection.execute(QUERY,params).fetchone()
    return None if row is None else row[0]


def populate(file,rows,indexed):
    start=time.perf_counter_ns()
    db=sqlite3.connect(file)
    db.execute("CREATE TABLE events(k INTEGER,valid INTEGER,tx INTEGER,v INTEGER)")
    db.executemany("INSERT INTO events VALUES(?,?,?,?)",rows)
    if indexed:
        db.execute("CREATE INDEX temporal ON events(k,valid DESC,tx DESC)")
    db.commit()
    return db,time.perf_counter_ns()-start


def history_rows(seed,size):
    rng=random.Random(seed)
    records=[]
    for tx in range(size):
        key=0 if rng.random()<.85 else rng.randrange(1,64)
        valid=tx if rng.random()<.8 else max(0,tx-rng.randrange(1,size//2))
        records.append((key,valid,tx,rng.randrange(1_000_000)))
    boundary=size*3//4
    queries=[(0 if rng.random()<.85 else rng.randrange(1,64),rng.randrange(boundary,size),
              rng.randrange(boundary,size)) for _ in range(192)]
    return records,boundary,queries


def temporal(seed,size,mode,folder):
    rows,boundary,queries=history_rows(seed,size)
    best={}
    start=time.perf_counter_ns()
    if mode=="compact":
        retained=[]
        for row in rows:
            k,valid,tx,v=row
            if valid>boundary or tx>boundary:
                retained.append(row)
            elif k not in best or row[1:3]>best[k][1:3]:
                best[k]=row
        records=retained+list(best.values())
    else:
        records=rows
    preparation=time.perf_counter_ns()-start
    file=Path(folder)/f"{mode}-{size}.db"
    db,build=populate(file,records,mode!="scan")
    start=time.perf_counter_ns()
    answers=[value(db,q) for q in queries]
    query_time=time.perf_counter_ns()-start
    pages=db.execute("PRAGMA page_count").fetchone()[0]
    page_size=db.execute("PRAGMA page_size").fetchone()[0]
    db.close()
    return dict(size=size,mode=mode,rows=len(records),domain_min=boundary,answers=answers,
                prep_ns=preparation,build_ns=build,query_ns=query_time,allocated_bytes=pages*page_size,
                file_bytes=file.stat().st_size,peak_parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run(seed,arm):
    with tempfile.TemporaryDirectory() as folder:
        base=race(Path(folder)/"concurrent.db","atomic" if arm=="candidate" else "split")
        hist=[temporal(seed,size,mode,folder) for size in (8192,65536)
              for mode in (("index","compact") if arm=="candidate" else ("scan",))]
    return dict(race=base,history=hist)
