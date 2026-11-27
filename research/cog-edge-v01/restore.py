"""Reassemble exact original local evidence, verify every member, then extract."""
import argparse, hashlib, pathlib, tempfile, zipfile
ROOT=pathlib.Path(__file__).resolve().parent
SHA="3d1bc2eeb7efec14a5b2e5c9c08cf8dbb4b65d0171028b7541ca810d0c67f768"
def check(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();out=pathlib.Path(a.out);assert not out.exists(),"output already exists"
 parts=sorted((ROOT/"archive-parts").glob("*.zip.part"));assert [x.name for x in parts]==[f"part-{i:03d}.zip.part" for i in range(10)]
 raw=b"".join(x.read_bytes() for x in parts);assert len(raw)==3721355 and check(raw)==SHA
 with tempfile.TemporaryDirectory() as td:
  archive=pathlib.Path(td)/"evidence.zip";archive.write_bytes(raw)
  with zipfile.ZipFile(archive) as z:
   names=z.namelist();assert len(names)==149 and len(set(names))==149
   manifest=dict(line.split("  ",1)[::-1] for line in z.read("SHA256SUMS.txt").decode().splitlines())
   assert set(names)==set(manifest)|{"SHA256SUMS.txt"} and len(manifest)==148
   for name in names:
    q=pathlib.PurePosixPath(name);assert not q.is_absolute() and ".." not in q.parts
    assert not (z.getinfo(name).external_attr>>16)&0o170000==0o120000
    if name in manifest:assert check(z.read(name))==manifest[name],name
   z.extractall(out)
 print("verified 149 files, archive SHA256",SHA)
if __name__=="__main__":main()
