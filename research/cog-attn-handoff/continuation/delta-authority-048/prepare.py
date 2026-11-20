"""Extract only the current user turn, immediately prior system turn and state diff."""
import hashlib
import json
import sys
from pathlib import Path

source=Path(sys.argv[1])
root=Path(__file__).resolve().parent/"recipe"
areas={"","centre","east","north","south","west","dontcare"}
data={}
hashes={}
for split in ("train","dev","test"):
    rows=[]
    for file in sorted((source/split).glob("dialogues_*.json")):
        blob=file.read_bytes()
        hashes[f"{split}/{file.name}"]=hashlib.sha256(blob).hexdigest()
        for dialogue in json.loads(blob):
            if "hotel" not in dialogue["services"]:
                continue
            prev=""
            system=""
            for turn in dialogue["turns"]:
                if turn["speaker"]=="SYSTEM":
                    system=turn["utterance"]
                    continue
                annotations={key:value for frame in turn["frames"]
                             for key,value in frame.get("state",{}).get("slot_values",{}).items()}
                slot=annotations.get("hotel-area",[""])
                if len(slot)!=1 or slot[0] not in areas:
                    prev=None
                    continue
                target=slot[0]
                rows.append(dict(shard=file.stem,dialogue=dialogue["dialogue_id"],turn=turn["turn_id"],
                                 user=turn["utterance"],system=system,old=prev,state=target,
                                 delta="hold" if prev==target else target))
                prev=target
    data[split]=rows
(root/"turns.json").write_text(json.dumps(data,separators=(",",":")))
(root/"corpus-sha256.json").write_text(json.dumps(hashes,indent=2))
print({k:len(v) for k,v in data.items()})
