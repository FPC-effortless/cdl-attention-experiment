"""Materialize deterministic new synthetic roles, with actor/truth separation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .role_control import CELLS, COLORS, families, oracle, world

def save(dirpath, name, blob):
    payload = (json.dumps(blob,sort_keys=True,separators=(",",":"),allow_nan=False) + "\n").encode()
    with (dirpath / name).open("xb") as f:
        f.write(payload)
    return sha256(payload).hexdigest()

def build(out):
    out.mkdir(exist_ok=False)
    splits = families()
    train_pairs = {k[0] for k in splits["fit"]}
    fit = [
        {"prompts":[" ".join(p) for p in prompts], "answers":[oracle(p) for p in prompts]}
        for prompts in [[world(key,c) for c in CELLS] for key in splits["fit"]]
    ]
    # Keep the registered actor payload free of family keys, factor signs,
    # answers, and sibling color IDs. Private alignment belongs to the judge.
    manifest={}
    manifest["train.json"]=save(out, "train.json",{
        "schema":"cog-comm-role-001.train.v1","role":"TRAIN","families":fit})
    for split in ("fit","control"):
        actor, judge = [], {"panel":split,"families":[]}
        for i,key in enumerate(splits[split]):
            batch = [world(key,c) for c in CELLS]
            actor.extend(" ".join(p) for p in batch)
            judge["families"].append(dict(
                rows=list(range(8*i,8*i+8)), key=key, answers=[oracle(p) for p in batch],
                familiar_name_pair=key[0] in train_pairs))
        manifest[split+"_actor.json"]=save(out,split+"_actor.json",{
            "schema":"cog-comm-role-001.actor.v1","prompts":actor})
        manifest[split+"_judge.json"]=save(out,split+"_judge.json",judge)
    save(out,"manifest.json",manifest)
    return manifest

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    print(json.dumps(build(ap.parse_args().out),sort_keys=True))
