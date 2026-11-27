"""Fail closed when original archives, manifests, weights or data are absent."""
from pathlib import Path
import hashlib,zipfile,json,sys,shutil
P=Path(__file__).resolve().parent
EXPECTED={
 "COG-DRONE-001-evidence.zip":("46c0c7b41e76b4f561df18a76da7dd74b3ed49ec76743725884932de778d675b",42,30),
 "COG-DRONE-002-006-evidence.zip":("80ac5e498a3927ba1c07deb7629e11ce703e79b4eebb8ef3e3be485063754ad2",97,81),
}
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    receipts=[];errors=[]
    root=P/"original"
    root.mkdir(exist_ok=True)
    for name,(sha,count,weights) in EXPECTED.items():
        try:
            p=P/"archives"/name
            assert p.is_file(),f"ORIGINAL_BYTES_MISSING: {p}"
            assert digest(p.read_bytes())==sha,"archive SHA256 mismatch"
            with zipfile.ZipFile(p) as z:
                names=z.namelist(); assert len(names)==count and len(set(names))==count
                assert sum(n.endswith(".pt") for n in names)==weights
                assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
                mf,= [n for n in names if n.endswith("/SHA256SUMS.txt")]
                base=mf.rsplit("/",1)[0]
                expected={base+"/"+line.split(None,1)[1]:line.split(None,1)[0] for line in z.read(mf).decode().splitlines()}
                assert set(expected)==set(names)-{mf}
                assert all(digest(z.read(n))==h for n,h in expected.items())
                z.extractall(root)
            receipts.append(dict(archive=name,sha256=sha,files=count,checkpoints=weights))
        except Exception as e:errors.append(dict(archive=name,error=repr(e)))
    result=dict(receipts=receipts,errors=errors)
    (P/"custody-result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if errors:sys.exit(1)
if __name__=="__main__":main()
