"""Deterministic clarification policies. Replies are tagged with a root and an epoch."""
import math
import random
from collections import defaultdict

import numpy as np

ACTIONS=np.array([(o,s,d) for o in range(4) for s in range(3) for d in range(3) if s!=d])
PROFILES=("exact","noisy","shared","hidden")
MODES={
    "control":("single","two","staged_labels","quorum_labels","verified"),
    "candidate":("staged_roots","quorum_roots"),
}
CURRENT=2


def reply_tape(seed,index,profile,truth):
    """Matched exogenous views; an indeterminate command has no hidden intended action."""
    rng=random.Random(seed*1000003+index*7919)
    corrupt=rng.random()<.15 and profile in ("shared","hidden")
    alternative=[rng.randrange(4),*rng.sample(range(3),2)]
    provider=[]
    for service in range(7):
        root=0 if profile=="shared" and service<4 else service
        wrong_snapshot=corrupt and service<4
        for slot,n in enumerate((4,3,3)):
            absent=rng.random()<.10 and profile!="exact"
            stale=rng.random()<.08 and profile!="exact"
            wrong=rng.random()<.05 and profile!="exact"
            if absent:
                status,val="missing",None
            elif truth is None:
                status,val=("answer",alternative[slot]) if wrong_snapshot or wrong else ("unresolved",None)
            else:
                val=truth[slot]
                if wrong_snapshot and slot:
                    val=truth[3-slot]
                elif wrong:
                    val=rng.choice([x for x in range(n) if x!=val])
                status="answer"
            provider.append(dict(service=service,root=root,slot=slot,status=status,value=val,
                                 epoch=CURRENT-1 if stale else CURRENT))
    return provider


def order(p,slots):
    entropy=[]
    for slot in slots:
        counts=np.bincount(ACTIONS[:,slot],weights=p,minlength=4 if slot==0 else 3)
        entropy.append(-sum(q*math.log(q) for q in counts if q>0))
    return [slot for _,slot in sorted(zip(entropy,slots),reverse=True)]


def normalize(weights):
    s=weights.sum()
    return weights/s if s else weights


class Dialogue:
    def __init__(self,tape,group):
        self.tape=tape
        self.group=group
        self.events=[]
        self.used=set()
        self.root_values=defaultdict(dict)
        self.posterior=None
        self.stop=None

    def next_reply(self,slot):
        # Prefer genuinely new groups when this policy consumes provenance.
        for response in self.tape:
            service,root=response["service"],response["root"] if self.group else response["service"]
            if response["slot"]==slot and (slot,service) not in self.used and root not in self.root_values[slot]:
                self.used.add((slot,service))
                self.events.append(response)
                return response,root
        return None,None

    def result(self,action,status):
        return dict(action=None if action is None else list(map(int,action)),reason=status,events=self.events)


def unguarded(prior,tape,mode):
    dialogue=Dialogue(tape,False)
    slots=order(prior,(0,1,2))[:1] if mode=="single" else (1,2)
    post=prior.copy()
    for slot in slots:
        reply,_=dialogue.next_reply(slot)
        if reply["status"]!="answer":
            return dialogue.result(None,reply["status"])
        post*=ACTIONS[:,slot]==reply["value"]
    return dialogue.result(ACTIONS[post.argmax()] if post.sum()>0 else None,
                           "conditioned" if post.sum()>0 else "incompatible")


def quorums(dialogue,slot):
    counts=np.bincount(list(dialogue.root_values[slot].values()),minlength=3)
    return [int(i) for i,n in enumerate(counts) if n>=2]


def guarded(prior,tape,mode,cutoff):
    quorum=mode.startswith("quorum")
    dialogue=Dialogue(tape,mode.endswith("roots"))
    post=normalize(np.sqrt(prior))
    op_prior=np.bincount(ACTIONS[:,0],weights=prior,minlength=4)
    limit=8 if quorum else 7
    blocked=set()
    for _ in range(limit):
        q={slot:quorums(dialogue,slot) for slot in (1,2)}
        if quorum and any(len(v)>1 for v in q.values()):
            return dialogue.result(None,"conflicting_quorums")
        unresolved=[slot for slot in (1,2) if slot not in blocked and (not quorum or not q[slot])]
        priorities=order(post,unresolved)
        # In the staged arm, see at least one provenance-bound response for each argument.
        if not quorum:
            priorities.sort(key=lambda slot:bool(dialogue.root_values[slot]))
        reply=None
        while priorities and reply is None:
            slot=priorities.pop(0)
            reply,root=dialogue.next_reply(slot)
            if reply is None:
                blocked.add(slot)
        if reply is None:
            return dialogue.result(None,"no_sources")
        if reply["epoch"]!=CURRENT:
            continue
        if reply["status"]=="unresolved":
            return dialogue.result(None,"authority_unresolved")
        if reply["status"]!="answer":
            continue
        val=reply["value"]
        dialogue.root_values[slot][root]=val
        post=normalize(post*np.where(ACTIONS[:,slot]==val,.95,.025))
        if not quorum and all(dialogue.root_values[k] for k in (1,2)) and post.max()>=cutoff:
            return dialogue.result(ACTIONS[post.argmax()],"posterior")
        if quorum:
            q={slot:quorums(dialogue,slot) for slot in (1,2)}
            if all(len(v)==1 for v in q.values()):
                src,dst=q[1][0],q[2][0]
                if src==dst or op_prior.max()<.99:
                    return dialogue.result(None,"unsupported_binding")
                return dialogue.result((op_prior.argmax(),src,dst),"quorum")
    return dialogue.result(None,"budget")


def score(prior,tape,mode,cutoff=1.01,truth=None):
    if mode=="verified":
        return dict(action=truth,events=[],reason="verified_fixture" if truth is not None else "unresolved")
    if mode in ("single","two"):
        return unguarded(prior,tape,mode)
    return guarded(prior,tape,mode,cutoff)


def adjust(action,state):
    if action is None:
        return state
    op,src,dst=action
    a,b=state[src],state[dst]
    result=state.copy()
    result[dst]=[a,(a+b)%8,(b-a)%8,max(a,b)][op]
    return result
