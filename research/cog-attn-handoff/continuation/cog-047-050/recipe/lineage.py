"""Bind acquisition ancestry and response fields to a trusted issuer."""
import hashlib
import hmac
import json
from dataclasses import asdict,dataclass

KEY=b"public-fixture-key-for-local-sealed-channel"


@dataclass(frozen=True)
class Snapshot:
    name: str
    epoch: int
    acquisition: str
    parents: tuple[str,...]


def signed(body):
    data=json.dumps(body,sort_keys=True,separators=(",",":")).encode()
    return hmac.new(KEY,data,hashlib.sha256).hexdigest()


class Book:
    def __init__(self,epoch,nodes):
        self.epoch=epoch
        self.nodes={}
        for node,signature in nodes:
            if not hmac.compare_digest(signed(asdict(node)),signature):
                raise ValueError("bad snapshot authentication")
            if node.name in self.nodes:
                raise ValueError("duplicate snapshot identifier")
            self.nodes[node.name]=node
        self.memo={}

    def roots(self,name,path=frozenset()):
        if name in path or name not in self.nodes:
            raise ValueError("cycle or missing provenance")
        if name in self.memo:
            return self.memo[name]
        node=self.nodes[name]
        if node.epoch!=self.epoch:
            raise ValueError("stale ancestry")
        if node.parents:
            if node.acquisition:
                raise ValueError("derived snapshot claims acquisition")
            roots=frozenset().union(*(self.roots(parent,path|{name}) for parent in node.parents))
        else:
            if not node.acquisition:
                raise ValueError("unidentified acquisition")
            roots=frozenset({node.acquisition})
        self.memo[name]=roots
        return roots

    def verify(self,receipt,signature,command):
        if not hmac.compare_digest(signed(receipt),signature):
            return None
        if receipt["epoch"]!=self.epoch or receipt["command"]!=command or receipt["role"] not in (1,2):
            return None
        if receipt["value"] not in range(3):
            return None
        try:
            return self.roots(receipt["snapshot"])
        except ValueError:
            return None


def authorize(book,receipts,command,mode="lineage"):
    by_role={1:[],2:[]}
    for body,sig in receipts:
        origins=book.verify(body,sig,command)
        if origins is not None:
            group=origins if mode=="lineage" else frozenset({body["snapshot"]})
            by_role[body["role"]].append((body["value"],group))
    result=[]
    for records in by_role.values():
        proven={a for i,(a,r) in enumerate(records) for b,s in records[i+1:] if a==b and not(r&s)}
        if len(proven)!=1:
            return None
        result.append(next(iter(proven)))
    return tuple(result) if result[0]!=result[1] else None


def registry(epoch,profile):
    # Two mirror identifiers descend through different nodes to one acquisition.
    roots=[Snapshot(f"input{i}",epoch,f"physical-{i}",()) for i in range(4)]
    mirrors=[Snapshot("mirror0",epoch,"",("input0",)),Snapshot("mirror1",epoch,"",("input0",)),
             Snapshot("mixed",epoch,"",("mirror0","input2")),Snapshot("mirror2",epoch,"",("input2",)),
             Snapshot("mirror3",epoch,"",("input3",))]
    if profile=="incomplete":
        roots=roots[1:]
    if profile=="concealed":
        mirrors[1]=Snapshot("mirror1",epoch,"",("input1",))
    if profile=="cycle":
        mirrors[0]=Snapshot("mirror0",epoch,"",("mirror1",))
        mirrors[1]=Snapshot("mirror1",epoch,"",("mirror0",))
    return [(x,signed(asdict(x))) for x in roots+mirrors]


def receipt(command,role,value,snapshot,epoch):
    info=dict(command=command,role=role,value=value,snapshot=snapshot,epoch=epoch)
    return info,signed(info)


def experiment():
    results=[]
    for epoch in (1,2,3,4):
        for profile in ("shared","incomplete","concealed","cycle","unrelated"):
            for mode in ("names","lineage"):
                book=Book(epoch,registry(epoch,profile))
                command=f"command-{epoch}"
                submissions=[receipt(command,role,val,snapshot,epoch)
                              for role,val in ((1,1),(2,0)) for snapshot in ("mirror0","mirror1")]
                if profile=="unrelated":
                    submissions=[receipt(command,role,val,snapshot,epoch)
                                 for role,val in ((1,0),(2,1)) for snapshot in ("mirror2","mirror3")]
                stale=receipt(command,1,0,"mirror2",epoch-1)
                wrong_command=receipt(command+"-other",2,2,"mirror3",epoch)
                tampered,sig=receipt(command,2,1,"mirror2",epoch)
                tampered["value"]=2
                assert book.verify(*stale,command) is None
                assert book.verify(*wrong_command,command) is None
                assert book.verify(tampered,sig,command) is None
                assert authorize(book,[receipt(command,r,v,p,epoch)
                                       for r,v in ((1,0),(2,1)) for p in ("mixed","mirror2")],command,"lineage") is None
                results.append(dict(epoch=epoch,profile=profile,mode=mode,
                                    admitted=authorize(book,submissions,command,mode)))
    return results
