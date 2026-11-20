"""Verify evidence at origin and distinguish identity from acquisition independence."""
import hashlib
import json
import random
from dataclasses import asdict,dataclass,replace

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":")).encode()


def key(label):
    # Deterministic keys are used only for reproducible lab fixtures.
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def sign(private,obj):
    return private.sign(canonical(obj))


def valid(public,signature,obj):
    try:
        public.verify(signature,canonical(obj))
    except InvalidSignature:
        return False
    return True


@dataclass(frozen=True)
class Grant:
    issuer: str
    family: str
    public_key: str
    era: int


class Authority:
    def __init__(self,seed):
        self.private=key(f"lab-ca-{seed}")
        self.files={}
        self.serial=0

    def enroll(self,issuer,family,era):
        private=key(f"{issuer}:{family}:{era}:{self.serial}")
        self.serial+=1
        self.files[issuer]=private
        grant=Grant(issuer,family,private.public_key().public_bytes_raw().hex(),era)
        return grant,sign(self.private,asdict(grant))

    def observe(self,grant,grant_sig,challenge,role,value,snapshot):
        body=dict(issuer=grant.issuer,**challenge,role=role,value=value,snapshot=snapshot)
        return dict(grant=grant,grant_sig=grant_sig,body=body,
                    response_sig=sign(self.files[grant.issuer],body),service=grant.issuer)


class Verifier:
    def __init__(self,ca):
        self.ca=ca
        self.used=set()

    def inspect(self,reply,challenge):
        body,grant=reply["body"],reply["grant"]
        if not valid(self.ca,reply["grant_sig"],asdict(grant)):
            return None
        if grant.era!=challenge["era"] or body["issuer"]!=grant.issuer:
            return None
        if any(body.get(field)!=value for field,value in challenge.items()):
            return None
        if body["role"] not in (1,2) or body["value"] not in range(3):
            return None
        public=Ed25519PublicKey.from_public_bytes(bytes.fromhex(grant.public_key))
        if not valid(public,reply["response_sig"],body):
            return None
        return grant.family

    def authorize(self,replies,challenge,policy):
        if challenge["nonce"] in self.used:
            return None
        fields={1:[],2:[]}
        for reply in replies:
            body=reply["body"]
            family=self.inspect(reply,challenge) if policy!="labels" else reply["service"]
            if family is None:
                continue
            discriminator=family if policy in ("scoped","digest") else reply["service"]
            fields[body["role"]].append((body["value"],discriminator,body["snapshot"]))
        result=[]
        for evidence in fields.values():
            eligible={a for i,(a,group,snapshot) in enumerate(evidence)
                      for b,h,other in evidence[i+1:] if a==b and group!=h and
                      (policy!="digest" or snapshot!=other)}
            if len(eligible)!=1:
                return None
            result.append(next(iter(eligible)))
        if result[0]==result[1]:
            return None
        self.used.add(challenge["nonce"])
        return result


PROFILES=("independent","unsigned_alias","modified_value","replay_nonce","wrong_origin_key","forged_grant",
          "enrolled_mirrors","copied_snapshot","equal_observations","hidden_common")


def fixture(authority,seed,index,profile):
    challenge=dict(era=3,command=f"command-{seed}-{index}",nonce=f"challenge-{seed}-{index}")
    a=authority.enroll("mirror-a","physical-1",3)
    b=authority.enroll("mirror-b","physical-1" if profile=="enrolled_mirrors" else "physical-2",3)
    names=("mirror-a","mirror-b")
    expected=[index%3,(index+1)%3]
    wrong=list(reversed(expected))
    authentic=profile in ("independent","equal_observations")
    values=expected if authentic or profile in ("modified_value","replay_nonce","wrong_origin_key","forged_grant") else wrong
    reply_challenge={**challenge,"nonce":challenge["nonce"]+"-previous"} if profile=="replay_nonce" else challenge
    shared_digest=profile in ("copied_snapshot","equal_observations")
    items=[]
    for k,source in enumerate((a,b)):
        number=0 if shared_digest else k
        digest=hashlib.sha256(f"snapshot-{seed}-{index}-{number}".encode()).hexdigest()
        for slot,value in zip((1,2),values,strict=True):
            item=authority.observe(*source,reply_challenge,slot,value,digest)
            item["service"]=names[k]
            if profile=="modified_value":
                item["body"]["value"]=wrong[slot-1]
            if profile=="wrong_origin_key":
                item["response_sig"]=sign(key("unauthorized-agent"),item["body"])
            if profile=="forged_grant":
                item["grant"]=replace(item["grant"],family="arbitrary-claimed-family")
            if profile=="unsigned_alias":
                item["grant_sig"]=b""
            items.append(item)
    return challenge,items,expected


def spot_audit(seed,period):
    rng=random.Random(seed+1933)
    rows=[]
    for trial in range(24):
        change=rng.randrange(9,55)
        count=queries=0
        detected=None
        for i in range(64):
            if period and i%period==0:
                queries+=1
                # The designated control observes the true binding directly.
                if i>=change and detected is None:
                    detected=i
            if i>=change and detected is None:
                count+=1
        rows.append(dict(trial=trial,shift=change,detected=detected,requests=queries,wrong_before_freeze=count))
    return rows


def run(seed,arm):
    policies=("scoped","digest") if arm=="candidate" else ("labels","signed")
    results=[]
    for profile in PROFILES:
        for index in range(64):
            authority=Authority(seed)
            challenge,replies,truth=fixture(authority,seed,index,profile)
            for policy in policies:
                inst=Verifier(authority.private.public_key())
                value=inst.authorize(replies,challenge,policy)
                if value is not None:
                    assert inst.authorize(replies,challenge,policy) is None
                results.append(dict(profile=profile,index=index,policy=policy,expected=truth,admitted=value))
    audits={str(period):spot_audit(seed,period) for period in (0,4,8,16)}
    return dict(seed=seed,results=results,spot=audits)
