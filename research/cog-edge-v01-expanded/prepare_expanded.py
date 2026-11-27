"""Use a pinned, connector-read ready-file manifest instead of anonymous tree API."""
import argparse,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--restored-source",required=True);p.add_argument("--archive",required=True);p.add_argument("--out",required=True)
a=p.parse_args();sys.path.insert(0,a.restored_source);import prepare
manifest=pathlib.Path(__file__).with_name("source-tree.json").read_bytes();meta=json.loads(manifest)
assert meta["revision"]==prepare.REV and not meta["truncated"] and len(meta["tree"])==277
assert len({x["path"] for x in meta["tree"]})==277
original_get=prepare.get
expected=f"https://api.github.com/repos/{prepare.REPO}/git/trees/{prepare.REV}?recursive=1"
def get(url):return manifest if url==expected else original_get(url)
prepare.get=get
arrays,record=prepare.expanded(a.archive);record["pinned_tree_manifest_sha256"]=prepare.sha(manifest)
path=pathlib.Path(a.out);np.savez(path,**arrays);path.with_suffix(".json").write_text(json.dumps(record,indent=2)+"\n")
print("admitted logs",len(arrays),"splits",{g:{k:len(v) for k,v in s.items()} for g,s in record["splits"].items()})
