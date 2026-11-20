"""Transactional grounding with a separate, deterministic clarification process."""
import hashlib
import json
import multiprocessing
import os
import resource
import random
import socket
import statistics
import struct
import sys
import time
from pathlib import Path
from workload import CHECKPOINT,LENGTH,PublicEvidenceReferencePolicy,features,np,predict,setup,trace,world
from attention import Decoder,PrefixDecoder,torch

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)


def retained_python_size(obj, seen=None):
    if seen is None:
        seen=set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size=sys.getsizeof(obj)
    if isinstance(obj,dict):
        size+=sum(retained_python_size(k,seen)+retained_python_size(v,seen) for k,v in obj.items())
    elif isinstance(obj,(list,tuple,set)):
        size+=sum(retained_python_size(v,seen) for v in obj)
    elif hasattr(obj,"model_dump"):
        size+=retained_python_size(obj.__dict__,seen)
    return size


def memory_kib(pid):
    samples={}
    for filename,fields in (("status",("VmHWM",)),("smaps_rollup",("Rss","Pss"))):
        for line in Path(f"/proc/{pid}/{filename}").read_text().splitlines():
            key,_,value=line.partition(":")
            if key in fields:
                samples[key]=int(value.split()[0])
    return samples


def responder(channel,bank):
    answers=dict(bank)
    while True:
        packet=channel.recv(4096)
        if not packet:
            break
        channel.sendall(struct.pack("!H",answers[packet.decode()]))
    channel.close()


def mutation(action,state):
    op,pair=divmod(action,1024)
    src,dst=divmod(pair,32)
    return dst,(state[src]+op*state[dst])%4


def execute(family,gate,ops,payloads,weights,vocab,bank,model,channel):
    start=time.perf_counter_ns()
    policy=PublicEvidenceReferencePolicy()
    kv=PrefixDecoder(model) if model is not None else None
    compiled,answers={},{}
    state=[i%4 for i in range(32)]
    reads,actions,rtts=[],[],[]
    rollbacks=0
    for kind,key,_ in ops:
        if kind=="read":
            if key not in answers:
                answers[key]=policy(payloads[key])
            reads.append(answers[key])
            continue
        command=bank[key][0]
        if command in compiled:
            chosen=compiled[command]
            dst,value=mutation(chosen,state)
        else:
            x,_=features([command],vocab)
            if family=="transformer":
                logits=kv(x[0].tolist())
                s,d,o=[v.softmax(-1).numpy() for v in logits.split((32,32,2),dim=-1)]
            else:
                s,d,o=predict(weights,x)
            chosen=int(s.argmax()*32+d.argmax()+o.argmax()*1024)
            confidence=float(s.max()*d.max()*o.max())
            if gate>.8:
                dst,value=mutation(chosen,state)  # private candidate write
            if confidence<gate:
                before=time.perf_counter_ns()
                channel.sendall(command.encode())
                confirmed=struct.unpack("!H",channel.recv(2))[0]
                rtts.append(time.perf_counter_ns()-before)
                if confirmed!=chosen:
                    rollbacks+=int(gate>.8)
                chosen=confirmed
                compiled[command]=confirmed  # Confirmed binding, valid for this fixed semantic epoch.
            else:
                compiled[command]=chosen
            # Nothing becomes visible until verification completes.
            if gate<=.8 or confidence<gate:
                dst,value=mutation(chosen,state)
        state[dst]=value
        fields=payloads[dst]["records"][-2]["fields"]
        fields[next(iter(fields))]=value
        answers.pop(dst,None)
        actions.append(chosen)
    elapsed=time.perf_counter_ns()-start
    return dict(ns=elapsed,memory=dict(cmd_cache_bytes=retained_python_size(compiled),
                            answer_cache_bytes=retained_python_size(answers),
                            kv_tensor_bytes=sum(t.numel()*t.element_size() for pair in kv.cache.values() for t in pair) if kv else 0),
                rtts=rtts,rollbacks=rollbacks,actions=actions,
                kv_hits=kv.hits if kv else 0,reads=[r.model_dump(mode="json") for r in reads])


def main():
    seed=int(os.environ["TRAINLAB_SEED"])
    family="wr" if sys.argv[1]=="candidate" else "transformer"
    weights,vocab,bank=setup(seed)
    model=None
    if family=="transformer":
        model=Decoder(len(vocab)).eval()
        with np.load(ROOT/f'matched-decoder-{CHECKPOINT[seed]}.npz') as ck:
            model.load_state_dict({k:torch.tensor(ck[k]) for k in ck.files})
    if model is not None:
        weights=[]
    parent,child=socket.socketpair(type=socket.SOCK_SEQPACKET)
    process=multiprocessing.get_context("fork").Process(target=responder,args=(child,bank),daemon=True)
    process.start()
    child.close()
    try:
        result=execute(family,.995,trace(seed,50),world(seed),weights,vocab,bank,model,parent)
        data=dict(family=family,seed=seed,main=memory_kib(os.getpid()),responder=memory_kib(process.pid),
                  model_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()) if model is not None else sum(a.nbytes for a in weights),
                  common_teacher_arrays_loaded=sum(a.nbytes for a in weights) if model is not None else 0,
                  **result["memory"])
        print(json.dumps(data))
    finally:
        parent.close()
        process.terminate()
        process.join()

if __name__=="__main__":
    main()
