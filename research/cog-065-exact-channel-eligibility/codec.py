"""Public-codec feasibility mechanics only, not a campaign generator, learner, or scorer."""
from itertools import combinations
import re

CLAUSES = {
    "status": {"completed": "status is completed", "unknown": "status is unknown", "blocked": "status is blocked"},
    "proof": {"accepted": "execution proof is accepted", "absent": "execution proof is absent"},
    "authority": {"current": "authority for operation is current", "revoked": "authority for operation is revoked"},
    "source": {"verified": "record issuer is independently verified", "unverified": "record issuer is unverified"},
    "scope": {"matched": "record scope matches request", "mismatch": "record scope differs from request"},
    "freshness": {"current": "record version is current", "stale": "record version is stale"},
    "progress": {"started": "progress is started", "partial": "progress is partial", "none": "progress is none"},
}
ORDER = tuple(CLAUSES)
LEX = re.compile(r"[a-z]+(?: [a-z]+)*\Z")
EOS = "<eos>"

def normalize_receipt(record):
    """Two known schema shapes; no guessed values or unsupported observation time."""
    if type(record) is not dict or type(record.get("version")) is not int:
        raise ValueError("a known typed receipt version is required")
    if record["version"] == 1 and set(record) == {"version","subject","emitter","event","available"}:
        key = (record["subject"],record["emitter"],record["event"],record["available"])
    elif record["version"] == 2 and set(record) == {"version","scope","provenance","observation"}:
        scope=record["scope"];issuer=record["provenance"];obs=record["observation"]
        if not isinstance(obs,dict) or set(obs)!={"kind","tick"}:
            raise ValueError("invalid observation")
        key = (scope,issuer,obs["kind"],obs["tick"])
    else:
        raise ValueError("unknown version or unexpected receipt fields")
    subject,issuer,event,tick=key
    if not all(type(v) is str and LEX.fullmatch(v) for v in (subject,issuer,event)):
        raise ValueError("invalid canonical text")
    if type(tick) is not int or tick<0:
        raise ValueError("invalid observation tick")
    return {"scope":subject,"issuer":issuer,"kind":event,"available":tick}

def observed_prefix(records,cutoff):
    if type(cutoff) is not int or cutoff<0:
        raise ValueError("bad cutoff")
    known=[normalize_receipt(r) for r in records]
    return tuple(sorted((r for r in known if r["available"]<=cutoff),key=lambda r:r["available"]))

def message_length(message):
    if type(message) is not str:
        raise ValueError("message must be a string")
    if message==EOS:
        return 1
    if not message.endswith(" "+EOS) or not LEX.fullmatch(message[:-len(EOS)-1]):
        raise ValueError("invalid lexer or EOS")
    return len(message.split(" "))

def encode(evidence,fields,budget):
    if type(budget) is not int or budget<1:
        raise ValueError("invalid token budget")
    fields=tuple(fields)
    if len(set(fields)) != len(fields) or not set(fields)<=set(evidence):
        raise ValueError("illegal or duplicate emission")
    try:
        terms=[CLAUSES[k][evidence[k]] for k in ORDER if k in fields]
        if len(terms)!=len(fields):
            raise KeyError("unknown field")
    except KeyError as e:
        raise ValueError("unsupported known evidence") from e
    m=" ".join(terms+[EOS])
    if message_length(m)>budget:
        raise ValueError("token budget exceeded")
    return m

def parse(message,budget):
    """Small frozen English-like grammar. Receiver uses decoded facts, not latent IDs."""
    if type(budget) is not int or message_length(message)>budget:
        raise ValueError("bad budget")
    rem=message.split(" ")[:-1]
    facts={}
    last=-1
    while rem:
        current=[]
        for idx,key in enumerate(ORDER):
            for value,phrase in CLAUSES[key].items():
                words=phrase.split()
                if rem[:len(words)]==words:
                    current.append((idx,key,value,len(words)))
        if len(current)!=1 or current[0][0]<=last:
            raise ValueError("unknown, repeated or out-of-order field")
        last,key,value,n=current[0]
        facts[key]=value
        rem=rem[n:]
    return facts

def exact_compress(evidence, weights, budget):
    """Exact additive utility control, selecting only grounded facts. No held truth/context."""
    keys=[k for k in ORDER if k in evidence]
    if set(evidence)-set(keys) or set(weights)!=set(keys) or any(type(v) is not int or v<0 for v in weights.values()):
        raise ValueError("closed evidence and nonnegative integer weights required")
    best=None
    for n in range(len(keys)+1):
        for fields in combinations(keys,n):
            try:m=encode(evidence,fields,budget)
            except ValueError:continue
            rank=(-sum(weights[k] for k in fields),message_length(m),m)
            if best is None or rank<best:
                best=rank
    if best is None:raise ValueError("no legal message")
    return best[-1]

def exact_expected(evidence, hypotheses, utility, budget):
    """Finite exact public-prior control; private realized context is never an input.

    Each hypothesis is (integer prior weight, public hypothetical receiver state).
    utility takes parsed clause facts and one hypothesis and returns integer utility.
    Caller owns independence/lineage of priors and utility rules.
    """
    hyp=tuple(hypotheses)
    keys=[k for k in ORDER if k in evidence]
    if set(evidence)-set(keys) or not hyp or any(type(p) is not int or p<=0 for p,_ in hyp):
        raise ValueError("closed evidence and nonempty positive integer prior required")
    try:
        for k in keys: CLAUSES[k][evidence[k]]
    except KeyError as e:
        raise ValueError("invalid evidence value") from e
    best=None
    for n in range(len(keys)+1):
        for fields in combinations(keys,n):
            try:message=encode(evidence,fields,budget)
            except ValueError:continue
            clauses=parse(message,budget)
            value=[utility(clauses,s) for _,s in hyp]
            if any(type(v) is not int for v in value):
                raise ValueError("integer utilities required")
            rank=(-sum(p*v for (p,_),v in zip(hyp,value)),message_length(message),message)
            if best is None or rank<best:
                best=rank
    if best is None:raise ValueError("no legal message")
    return best[-1]
