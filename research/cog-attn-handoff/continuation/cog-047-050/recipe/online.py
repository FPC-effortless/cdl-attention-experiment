"""Run frozen neural text grounding and provenance verification inside a timed loop."""
import json
import multiprocessing as mp
import socket
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from binding import words_for_seed
from continual import effect
from fusion import ContextHead, Entailment, SEEDS, data_for, load_encoder, rows_for
from lineage import Book, authorize, receipt, registry
from state_store import World

ROOT=Path(__file__).resolve().parent


def listen(pipe,world_truth):
    with socket.socket() as server:
        server.bind(("127.0.0.1",0))
        server.listen()
        pipe.send(server.getsockname()[1])
        conn,_=server.accept()
        with conn,conn.makefile("rb") as incoming:
            for line in incoming:
                epoch,index=json.loads(line)
                action=world_truth[index]
                evidence=[]
                if action is not None:
                    for role in (1,2):
                        for origin in ("mirror0","mirror1","mirror2","mirror3"):
                            if epoch==2 and origin=="mirror3":
                                continue
                            value=action[3-role] if epoch and origin in ("mirror0","mirror1") else action[role]
                            evidence.append(receipt(str(index),role,value,origin,epoch))
                conn.sendall((json.dumps(evidence)+"\n").encode())


def compile_model():
    started=time.perf_counter_ns()
    nli=Entailment()
    encoder=load_encoder("pretrained")
    heads=[]
    for i,_seed in enumerate(SEEDS):
        head=ContextHead()
        with np.load(ROOT/f"minilm-head-{i}.npz") as data:
            head.load_state_dict({key:torch.tensor(v) for key,v in data.items()})
        heads.append(head.eval())
    return (nli,encoder,heads),time.perf_counter_ns()-started


def activate(raw,cohort,seed,setup,fuser,threshold):
    nli,encoder,head=setup
    tokens,names,*_=raw
    r=rows_for([raw],cohort,seed,2,nli,encoder,head)[0]
    confidence=float(fuser.predict_proba([r["features"]])[0,1])
    auto=r["local_contract"] or (r["blended"] if confidence>=threshold else None)
    return (r["blended"][0] if r["features"][8]>=.99 else None),auto


def evaluate(seed,setup,weights,threshold,mode):
    data=data_for(seed,2)
    frames=[(raw,cohort) for cohort in ("typed","implicit","correction","ambiguous") for raw in data[cohort][:3]]
    history=(frames[:2]+frames[3:5]+frames[6:8]+frames[9:11]+frames[:2]+frames[3:5])*4
    truth=[None if group=="ambiguous" else list(row[4:]) for row,group in history]
    durations={k:0 for k in ("inference","verification","clarification","repair","commit")}
    parent,child=mp.Pipe(False)
    server=mp.get_context("fork").Process(target=listen,args=(child,truth),daemon=True)
    start=time.perf_counter_ns()
    server.start()
    child.close()
    port=parent.recv()
    world=World()
    cold=time.perf_counter_ns()-start
    cache={}
    predictions=[]
    book=None
    state=dict.fromkeys({name for row,_ in history for name in row[1]},1)
    full=state.copy()
    with socket.create_connection(("127.0.0.1",port),timeout=10) as conn,conn.makefile("rb") as incoming:
        conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        timer=time.perf_counter_ns()
        for i,(row,cohort) in enumerate(history):
            if i%12==0:
                begin=time.perf_counter_ns()
                if i:
                    world.advance()
                # Drop old bindings instead of ever allowing an old epoch hit.
                cache.clear()
                book=Book(world.epoch(),registry(world.epoch(),"shared"))
                durations["repair"]+=time.perf_counter_ns()-begin
            tokens,names,*_=row
            epoch=world.epoch()
            key=(epoch,tuple(tokens))
            begin=time.perf_counter_ns()
            hit=key in cache
            if hit:
                op,auto=cache[key]
            else:
                op,auto=activate(row,cohort,seed,setup,weights,threshold)
                if mode=="cache" and auto is not None:
                    cache[key]=op,auto
            durations["inference"]+=time.perf_counter_ns()-begin
            begin=time.perf_counter_ns()
            conn.sendall((json.dumps([epoch,i])+"\n").encode())
            evidence=json.loads(incoming.readline())
            durations["verification" if auto is not None else "clarification"]+=time.perf_counter_ns()-begin
            begin=time.perf_counter_ns()
            binding=authorize(book,evidence,str(i))
            if binding and op is not None:
                action=[op,*binding]
                values=effect(action,[state[name] for name in names])
                state.update(zip(names,values))
                world.commit(str(i),epoch,state)
            else:
                action=None
            if epoch!=2 and truth[i] is not None:
                values=effect(truth[i],[full[name] for name in names])
                full.update(zip(names,values))
            durations["commit"]+=time.perf_counter_ns()-begin
            predictions.append(dict(i=i,epoch=epoch,cohort=cohort,truth=truth[i],committed=action,
                                    auto=auto,cache_hit=hit,exact_state=state==full))
        elapsed=time.perf_counter_ns()-timer
    server.join(10)
    assert server.exitcode==0
    world.db.close()
    return dict(seed=seed,mode=mode,cold_ns=cold,elapsed_ns=elapsed,stages_ns=durations,steps=predictions)


def run(path):
    start=time.perf_counter_ns()
    evidence=json.loads(Path(path).read_text())
    train=[r for r in evidence["rows"] if r["split"]==0]
    weights=LogisticRegression(max_iter=1000).fit([r["features"] for r in train],
                [r["blended"]==r["label"] for r in train])
    fit=time.perf_counter_ns()-start
    (nli,encoder,heads),cold=compile_model()
    results=[]
    for seed,head in zip(SEEDS,heads,strict=True):
        for mode in ("fresh","cache"):
            results.append(evaluate(seed,(nli,encoder,head),weights,evidence["threshold"],mode))
    return dict(encoder_load_ns=cold,fusion_fit_ns=fit,streambench=results)
