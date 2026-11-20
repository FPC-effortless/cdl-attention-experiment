import json
import sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from attestation import Verifier
from wire import authorize_bytes
vectors=json.loads((ROOT/"vectors.json").read_text())
results={}
for item in vectors:
    proto=Verifier(Ed25519PublicKey.from_public_bytes(bytes.fromhex(item["ca"])))
    message=json.dumps(item["replies"]).encode()
    out=authorize_bytes(proto,message,item["request"])
    accept=out is not None
    assert accept==(item["profile"] in ("independent","copied_snapshot","hidden_common"))
    if accept:
        assert authorize_bytes(proto,message,item["request"]) is None
    results[item["profile"]]=out
for payload in (b"[{}]",b'[{"grant":0,"grant":1}]',bytes([255]),b" "*65537,b"["*3000):
    assert authorize_bytes(proto,payload,item["request"]) is None
results["malformed_boundaries_rejected"]=5
(ROOT/"wire-check.json").write_text(json.dumps(results,indent=2))
