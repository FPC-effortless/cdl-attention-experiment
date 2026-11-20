"""Transactional grounding with a separate, deterministic clarification process."""
import hashlib
import json
import multiprocessing
import os
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
    return dict(ns=elapsed,rtts=rtts,rollbacks=rollbacks,actions=actions,
                kv_hits=kv.hits if kv else 0,reads=[r.model_dump(mode="json") for r in reads])


def main():
    seed=int(os.environ["TRAINLAB_SEED"])
    family="wr" if sys.argv[1]=="candidate" else "transformer"
    weights,vocab,bank=setup(seed)
    model=None
    if family=="transformer":
        model=Decoder(len(vocab)).eval()
        with np.load(ROOT/f'decoder-{CHECKPOINT[seed]}.npz') as ck:
            model.load_state_dict({k:torch.tensor(ck[k]) for k in ck.files})
    parent,child=socket.socketpair(type=socket.SOCK_SEQPACKET)
    process=multiprocessing.get_context("fork").Process(target=responder,args=(child,bank),daemon=True)
    process.start()
    child.close()
    rows=[]
    try:
        for ratio in (1,10,50,90):
            ops=trace(seed,ratio)
            gates=[.80,.995]+([2.] if family=="wr" else [])
            times={gate:[] for gate in gates}
            last={}
            rng=random.Random(ratio+seed)
            for _ in range(3):
                rng.shuffle(gates)
                for gate in gates:
                    cell=execute(family,gate,ops,world(seed),weights,vocab,bank,model,parent)
                    times[gate].append(cell.pop("ns"))
                    last[gate]=cell
            for gate in gates:
                rows.append(dict(family=family,mode=str(gate),ratio=ratio,times=times[gate],**last[gate]))
    finally:
        parent.close()
        process.terminate()
        process.join()
    out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    (out/"results.json").write_text(json.dumps(rows,separators=(",",":")))
    primary=next(r for r in rows if r['ratio']==10 and r['mode']=='0.8')
    (out/"metrics.json").write_text(json.dumps(dict(score=LENGTH*1e9/statistics.median(primary["times"]))))
    stable=sorted((r["mode"],r["ratio"],r["actions"],r["reads"]) for r in rows)
    (out/"responses.sha256").write_text(hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest())

if __name__=="__main__":
    main()
