"""Data provenance and sequence qualification on local scripted correction sessions."""
from dataclasses import asdict,dataclass
import hashlib,hmac,json,random
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from boundary import Authority,Ledger,OPERATIONS
from tracker import infer

@dataclass(frozen=True)
class Message:
    session:str
    principal:str
    sequence:int
    value:int
    revision:bool

def signed(message,key):
    body=json.dumps(asdict(message),sort_keys=True).encode()
    return message,hmac.digest(key,body,'sha256')

def source_filter(events,session,keys):
    latest={};result=[]
    for m,tag in events:
        if m.session!=session or m.principal not in keys:continue
        _,expected=signed(m,keys[m.principal])
        if not hmac.compare_digest(expected,tag) or m.sequence<=latest.get(m.principal,-1):continue
        if m.revision and m.principal!='requester':continue
        latest[m.principal]=m.sequence;result.append(m)
    return result

def fixture(seed):
    rng=random.Random(seed);k={s:rng.randbytes(32) for s in ('requester','mirror')}
    result=[]
    with TemporaryDirectory() as temp:
        key=rng.randbytes(32);issuer=Authority(key)
        stores={a:Ledger(Path(temp)/a,key) for a in ('claimed','signed')}
        for case in ('legitimate','mirror','late','mistaken','silence'):
            for intent in range(32):
                sid=case+str(intent);wrong=(intent+rng.randrange(1,32))%32
                def envelope(i,value,revision,origin='requester'):
                    return signed(Message(sid,'requester',i,value,revision),k[origin])
                events=[envelope(1,wrong if case in ('legitimate','late') else intent,False)]
                if case in ('legitimate','late'):events.append(envelope(3,intent,True))
                if case=='mirror':events.append(envelope(2,wrong,True,'mirror'))
                if case=='late':events.append(envelope(2,wrong,True))
                if case=='mistaken':events.append(envelope(2,wrong,True))
                for arm,store in stores.items():
                    ev=source_filter(events,sid,k) if arm=='signed' else [m for m,_ in events]
                    beliefs=[]
                    for m in ev:
                        p=np.full(8,.001);p[m.value//4]=.993
                        q=np.full(4,.001);q[m.value%4]=.997
                        beliefs.append((m.principal,m.revision,p,q))
                    a,t=infer(beliefs,'full');op=int(a.argmax());ticket=int(t.argmax())
                    proposal=store.propose(sid,ticket,OPERATIONS[op])
                    # Silence never supplies an action credential.
                    token=issuer.issue(proposal,case!='silence',True)
                    status=store.commit(proposal,token)
                    assert (status=='committed')==(case!='silence')
                    result.append([case,arm,intent,op*4+ticket,status,len(ev)])
        for db in stores.values():db.db.close()
    return result
if __name__=='__main__':
    p=Path(__file__).resolve().parent/'results/corrections.json'
    p.write_text(json.dumps({s:fixture(s) for s in (8231,8233,8237)},separators=(',',':')))
