"""Preserve the official dialogue split and save typed evaluation records."""
import hashlib
import json
import sys
from pathlib import Path

source=Path(sys.argv[1])
dest=Path(__file__).resolve().parent/"recipe"
supported={"","centre","east","north","south","west","dontcare"}
records={}
hashes={}
for split in ("train","dev","test"):
    files=sorted((source/split).glob("dialogues_*.json")) if split=="train" else [source/split/"dialogues_002.json"]
    output=[]
    for filename in files:
        raw=filename.read_bytes()
        hashes[f"{split}/{filename.name}"]=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        for dialogue in data:
            if "hotel" not in dialogue["services"]:
                continue
            history=[]
            previous=""
            for turn in dialogue["turns"]:
                speaker=turn["speaker"]
                text=turn["utterance"]
                current=" ".join([*(history[-1:] if history else []),text])
                history.append(text)
                if speaker!="USER":
                    continue
                slots={k:v for fr in turn["frames"] for k,v in fr.get("state",{}).get("slot_values",{}).items()}
                v=slots.get("hotel-area",[""])
                if len(v)!=1 or v[0] not in supported:
                    previous=None
                    continue
                v=v[0]
                output.append(dict(dialogue=dialogue["dialogue_id"],turn=turn["turn_id"],current=current,
                                   context=" [TURN] ".join(history[-7:]),target=v,
                                   action="hold" if v==previous else v))
                previous=v
    records[split]=output
(dest/"dialogue-expanded.json").write_text(json.dumps(records,separators=(",",":")))
(dest/"dialogue-original-shards.json").write_text(json.dumps(hashes,indent=2))
print({k:len(v) for k,v in records.items()})
