import hashlib
import json
import multiprocessing
import os
import socket
import sys
from pathlib import Path
from workload import features,np,predict,setup,trace,world,PublicEvidenceReferencePolicy

ARMS={"candidate":("traced","global","untracked_selector"),"control":("reuse","precheck","always")}
RATIOS=(10,50,90)


def group(command):
    return sum(command.split()[1].encode())%4


def change_at(write):
    # Three nonoverlapping phases in each block exercise independent transitions.
    return {8:"meaning",16:"inactive",24:"handoff",31:"race"}.get(write%32)


def keys(g,epochs):
    selector=f"selector-{g}"
    provider=epochs.get(selector,0)%2
    return selector,f"source-{g}-{provider}",provider


def changed_key(g,kind,epochs):
    selector,source,p=keys(g,epochs)
    if kind=="inactive":
        return f"source-{g}-{1-p}"
    return selector if kind=="handoff" else source


def responder(sock,bank):
    bindings=dict(bank)
    epochs={}
    while packet:=sock.recv(4096):
        cmd,arg=json.loads(packet)
        if cmd=="stop":
            break
        if cmd=="advance":
            epochs[arg]=epochs.get(arg,0)+1
            result=None
        else:
            # Capture the selector and selected provider through mediated reads.
            stamps={}
            def read(key):
                stamps[key]=epochs.get(key,0)
                return stamps[key]
            g=group(arg)
            p=read(f"selector-{g}")%2
            rev=read(f"source-{g}-{p}")
            result=[bindings[arg]^(1024*((p+rev)%2)),stamps]
        sock.sendall(json.dumps(result).encode())


def rpc(channel,op,arg):
    channel.sendall(json.dumps([op,arg]).encode())
    return json.loads(channel.recv(4096))


def execute(seed,ratio,mode,ops,weights,vocab,bank,channel):
    epochs={}
    generation=0
    cache={}
    payloads=world(seed)
    states=[i%4 for i in range(32)]
    policy=PublicEvidenceReferencePolicy()
    answers={}
    reads=[]
    records=[]
    def advance(g,kind):
        nonlocal generation
        key=changed_key(g,kind,epochs)
        epochs[key]=epochs.get(key,0)+1
        generation+=1
        rpc(channel,"advance",key)
    def valid(entry):
        if mode=="always":
            return False
        if mode=="reuse":
            return True
        if mode=="global":
            return entry[2]==generation
        return all(epochs.get(k,0)==v for k,v in entry[1].items())
    def confirm(command):
        ids,_=features([command],vocab)
        s,d,o=predict(weights,ids)
        proposal=int(o.argmax()*1024+s.argmax()*32+d.argmax())
        action,stamps=rpc(channel,"resolve",command)
        if mode=="untracked_selector":
            stamps={k:v for k,v in stamps.items() if k.startswith("source")}
        return (action,stamps,generation),int(proposal!=action)
    for kind,key,_ in ops:
        if kind=="read":
            if key not in answers:
                answers[key]=policy(payloads[key])
            reads.append(answers[key].model_dump(mode="json"))
            continue
        command=bank[key][0]
        g=group(command)
        shift=change_at(len(records)+1)
        if shift and shift!="race":
            advance(g,shift)
        entry=cache.get(command)
        calls=disagreements=0
        if entry is None or not valid(entry):
            entry,disagreements=confirm(command)
            calls=1
        if shift=="race":
            advance(g,shift)
        if mode not in ("precheck","reuse") and (mode=="always" and shift=="race" or mode!="always" and not valid(entry)):
            entry,mismatch=confirm(command)
            calls+=1
            disagreements+=mismatch
        cache[command]=entry
        action=entry[0]
        src,dst=action//32%32,action%32
        states[dst]=(states[src]+(action//1024)*states[dst])%4
        fields=payloads[dst]["records"][-2]["fields"]
        fields[next(iter(fields))]=states[dst]
        answers.pop(dst,None)
        records.append([action,calls,disagreements])
    return dict(mode=mode,ratio=ratio,writes=records,reads=reads)


if __name__=="__main__":
    seed=int(os.environ["TRAINLAB_SEED"])
    weights,vocab,bank=setup(seed)
    rows=[]
    for ratio in RATIOS:
        ops=trace(seed,ratio)
        for mode in ARMS[sys.argv[1]]:
            parent,child=socket.socketpair(type=socket.SOCK_SEQPACKET)
            proc=multiprocessing.get_context("fork").Process(target=responder,args=(child,bank))
            proc.start()
            child.close()
            try:
                rows.append(execute(seed,ratio,mode,ops,weights,vocab,bank,parent))
            finally:
                parent.sendall(b'["stop",null]')
                parent.close()
                proc.join(5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
    out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    payload=json.dumps(rows,separators=(",",":")).encode()
    (out/"results.json").write_bytes(payload)
    (out/"responses.sha256").write_text(hashlib.sha256(payload).hexdigest())
    primary=rows[0]
    # Primary measure is reuses per write, safety checked separately by replay.
    reused=sum(r[1]==0 for r in primary["writes"])
    (out/"metrics.json").write_text(json.dumps({"score":reused/len(primary["writes"])}))
