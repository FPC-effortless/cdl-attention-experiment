"""Bound and validate an untrusted serialized reply batch before signature checks."""
import json
import re

from attestation import Grant

HEX32=re.compile(r"[0-9a-f]{64}\Z")
HEX64=re.compile(r"[0-9a-f]{128}\Z")
FIELD_LIMIT=192
MAX_REPLIES=64
MAX_BYTES=65536


def no_duplicates(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:
            raise ValueError("duplicate field")
        obj[key]=value
    return obj


def parse(payload):
    if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:
        raise ValueError("bad wire size")
    rows=json.loads(payload,object_pairs_hook=no_duplicates)
    if not isinstance(rows,list) or len(rows)>MAX_REPLIES:
        raise ValueError("bad reply batch")
    output=[]
    for item in rows:
        if not isinstance(item,dict) or set(item)!={"body","grant","grant_sig","response_sig","service"}:
            raise ValueError("bad reply fields")
        b,g=item["body"],item["grant"]
        if not isinstance(b,dict) or set(b)!={"issuer","era","command","nonce","role","value","snapshot"}:
            raise ValueError("bad signed body")
        if not isinstance(g,dict) or set(g)!={"issuer","family","public_key","era"}:
            raise ValueError("bad certificate")
        for val in (b["issuer"],b["command"],b["nonce"],g["issuer"],g["family"],item["service"]):
            if not isinstance(val,str) or not 1<=len(val)<=FIELD_LIMIT:
                raise ValueError("bad identifier")
        for val in (b["era"],g["era"],b["role"],b["value"]):
            if type(val) is not int or not 0<=val<1_000_000_000:
                raise ValueError("bad scalar")
        for val,p in ((g["public_key"],HEX32),(b["snapshot"],HEX32),
                      (item["grant_sig"],HEX64),(item["response_sig"],HEX64)):
            if not isinstance(val,str) or p.fullmatch(val) is None:
                raise ValueError("bad binary field")
        output.append(dict(body=b,grant=Grant(**g),service=item["service"],
                           grant_sig=bytes.fromhex(item["grant_sig"]),response_sig=bytes.fromhex(item["response_sig"])))
    return output


def authorize_bytes(inst,payload,challenge,policy="scoped"):
    try:
        return inst.authorize(parse(payload),challenge,policy)
    except (ValueError,TypeError,RecursionError):
        return None
