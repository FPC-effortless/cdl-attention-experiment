"""Live loopback verification and atomic typed commits across repeated shifts."""
import json
import multiprocessing as mp
import socket
import time

import torch

from causal import Attention,DIMS,LENGTHS,Local,data,decode,fit
from lineage import Book,authorize,receipt,registry
from state_store import World

TURNS=48
COST_WEIGHT=12


def listen(pipe,delay,hidden):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1",0))
        sock.listen()
        pipe.send(sock.getsockname()[1])
        conn,_=sock.accept()
        with conn,conn.makefile("rb") as stream:
            for line in stream:
                epoch,key=json.loads(line)
                truth=hidden[int(key)]
                truth=(truth[0]^int(epoch%2==1),*truth[1:])
                profile=("clean","shared","missing","shared")[epoch]
                if delay:
                    time.sleep(delay)
                # Two mirrors with shared corrupt values, plus independent records.
                submissions=[]
                for slot in (1,2):
                    for provider in ("mirror0","mirror1","mirror2","mirror3"):
                        if profile=="missing" and provider=="mirror3":
                            continue
                        value=truth[3-slot] if provider in ("mirror0","mirror1") and profile!="clean" else truth[slot]
                        submissions.append(receipt(key,slot,value,provider,epoch))
                conn.sendall((json.dumps(submissions)+"\n").encode())


def effect(action,state):
    op,s,d=action
    value=state[s]
    old=state[d]
    return [v if i!=d else (value,(old+value)%8,(old-value)%8,max(old,value))[op]
            for i,v in enumerate(state)]


def runner(model,mode,seed,length,delay,fit_ns):
    start=time.perf_counter_ns()
    x,y=data(torch.Generator().manual_seed(seed+4300),TURNS,length)
    # Every arm sees the same typed active frame and shared archive. Cold-start
    # grows with requested length. A frozen shared prefix supports KV reuse.
    archive=x[:1,:-4].clone()
    cache=model.cache(archive) if mode=="full" else None
    cold=time.perf_counter_ns()-start
    prefix_bytes=sum(t.numel()*t.element_size() for t in cache) if cache else 0
    parity=0
    if cache is not None:
        test=torch.cat((archive,x[:1,-4:]),1)
        parity=float((model(test)-model.cached(x[:1,-4:],cache)).abs().max())
    parent,child=mp.Pipe(False)
    server=mp.get_context("fork").Process(target=listen,args=(child,delay,y.tolist()),daemon=True)
    start=time.perf_counter_ns()
    server.start()
    child.close()
    port=parent.recv()
    world=World()
    key_state=[1,4,6]
    expected=key_state.copy()
    times=dict(inference=0,clarification=0,verification=0,repair=0,commit=0)
    audit=[]
    cached_book=None
    cold+=time.perf_counter_ns()-start
    start_all=time.perf_counter_ns()
    with socket.create_connection(("127.0.0.1",port),timeout=5) as conn,conn.makefile("rb") as stream:
        conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        for i,(tokens,target) in enumerate(zip(x,y,strict=True)):
            if i%12==0:
                start=time.perf_counter_ns()
                if i:
                    world.advance()
                cached_book=Book(world.epoch(),registry(world.epoch(),"shared"))
                times["repair"]+=time.perf_counter_ns()-start
            epoch=world.epoch()
            gold=tuple(target.tolist())
            # A new public authority can invert the COPY/ADD orientation.
            # The same published correction is available to both candidates.
            gold=(gold[0]^int(epoch%2==1),*gold[1:])
            profile=("clean","shared","missing","shared")[epoch]
            before=time.perf_counter_ns()
            x_live=tokens[None,-4:]
            raw=decode(model.cached(x_live,cache) if mode in ("full","scoped") else model(x_live))
            raw=(raw[0]^int(epoch%2==1),*raw[1:])
            times["inference"]+=time.perf_counter_ns()-before
            verify=i%4!=0 or profile!="clean"
            action=raw if raw[1]!=raw[2] else None
            requests=0
            if verify:
                before=time.perf_counter_ns()
                wire=[epoch,str(i)]
                conn.sendall((json.dumps(wire)+"\n").encode())
                replies=json.loads(stream.readline())
                requests+=1
                times["clarification"]+=time.perf_counter_ns()-before
                before=time.perf_counter_ns()
                resolved=authorize(cached_book,replies,str(i))
                action=(raw[0],*resolved) if resolved else None
                times["verification"]+=time.perf_counter_ns()-before
            # Retry across one visible commit race at the first turn of a
            # changed segment. Record the rejection without allowing stale state.
            before=time.perf_counter_ns()
            race=i%12==0 and i>0
            if race:
                assert world.commit(f"race-{i}",epoch-1,{"cells":[0,0,0]})=="stale"
            if action:
                key_state=effect(action,key_state)
                status=world.commit(str(i),epoch,{"cells":key_state})
                assert status=="committed"
            times["commit"]+=time.perf_counter_ns()-before
            if profile!="missing":
                expected=effect(gold,expected)
            audit.append(dict(i=i,epoch=epoch,gold=gold,raw=raw,action=action,race=race,
                              request=requests,profile=profile,state=key_state.copy(),expected=expected.copy()))
    elapsed=time.perf_counter_ns()-start_all
    server.join(5)
    assert server.exitcode==0
    world.db.close()
    return dict(seed=seed,length=length,mode=mode,charge_ms=delay*1000,cold_ns=cold,elapsed_ns=elapsed,
                fit_ns=fit_ns,total_ns=fit_ns+cold+COST_WEIGHT*elapsed,kv_bytes=prefix_bytes,
                kv_parity=parity,stages_ns=times,steps=audit)


def run(seed,arm,folder):
    modes=("local",) if arm=="candidate" else ("full","scoped")
    results=[]
    for mode in modes:
        torch.manual_seed(seed)
        model=Local() if mode=="local" else Attention()
        fit_ns=fit(model,seed,scoped=mode=="scoped")
        torch.save(model.state_dict(),folder/f"{mode}.pt")
        model.eval()
        with torch.no_grad():
            for length in LENGTHS:
                for delay in (0,.002):
                    results.append(runner(model,mode,seed,length,delay,fit_ns))
    return results
