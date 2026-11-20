"""Selective typed contract for explicit hotel-area assertions."""
import itertools
import json
import re
from pathlib import Path

from state_store import World

ROOT=Path(__file__).resolve().parent
AREA=r"(?P<area>centre|center|downtown|north|south|east|west|northern|southern|eastern|western)"
FAMILIAR={
    "center":"centre","downtown":"centre","northern":"north","southern":"south","eastern":"east","western":"west",
}
KEEP=(
    r"(?:i am|i'm|im) (?:also )?looking for",
    r"i (?:would )?(?:also )?(?:like|need|want)",
    r"(?:could|can) you (?:help me )?(?:find|book)(?: me)?",
    r"(?:i'd|id) (?:also )?like",
    r"(?:please )?(?:find|book) me",
)
POLITE=r"(?:please|thanks|thank you|if possible)"
MOD=r"(?:some|a|an|the)(?:\s+[a-z0-9-]+){0,5}\s+"
TYPE=r"(?:hotel|guest house|guesthouse|place to stay)"
PREFIX=rf"(?:(?:hi|hello|yes|thanks|thank you)[, ]+)?(?:{'|'.join(KEEP)})\s+"
TAIL=rf"(?:\s+(?:side|part|area|end)(?:\s+of\s+(?:town|the city|cambridge))?|\s+of\s+(?:town|cambridge))?"
pattern=re.compile(rf"{PREFIX}{MOD}{TYPE}\s+(?:in|on|near)\s+(?:the\s+)?{AREA}{TAIL}(?:[ ,]+{POLITE})?[.?!]*",re.I)
STOP=re.compile(r"\b(?:or|not|any|but|outside|unless|previous|restaurant|train)\b",re.I)


def candidate(text):
    if STOP.search(text):
        return None
    m=pattern.fullmatch(text.strip())
    if m is None:
        return None
    area=m["area"].lower()
    return FAMILIAR.get(area,area)


def run_policy(raw,rows,gate,save=False):
    state={}
    admitted=correct=0
    out=[]
    world=World() if save else None
    t,n,p=gate
    for sample,row in zip(raw,rows,strict=True):
        action=candidate(sample["user"])
        if action is not None and (t or n or p):
            if row["candidate"]!=action or row["p"]<p or row["delta"]<t or row["nli"]<n:
                action=None
        if action is not None and action!=state.get(sample["dialogue"],""):
            admitted+=1
            correct+=action==sample["delta"]
            state[sample["dialogue"]]=action
            if save:
                assert world.commit(f"{sample['dialogue']}:{sample['turn']}",0,{sample["dialogue"]:action})=="committed"
        else:
            action=None
        if save:
            out.append(dict(dialogue=sample["dialogue"],turn=sample["turn"],admitted=action,
                            truth=sample["delta"],target=sample["state"],predicted=state.get(sample["dialogue"],"")))
    if save:
        assert world.state()==state
        world.db.close()
    return dict(admitted=admitted,correct=correct,wrong=admitted-correct,updates=sum(r["delta"]!="hold" for r in raw),
                rows=out)


def main():
    data=json.loads((ROOT/"turns.json").read_text())
    score=json.loads((ROOT.parent/"delta.json").read_text())["scores"]
    gates=list(itertools.product((0.,.8,.9,.95,.98,.99),(0.,.5,.8,.95),(0.,.10,.25,.50)))
    candidates=[]
    for gate in gates:
        trial=run_policy(data["dev"],score["dev"],gate)
        if trial["admitted"]>=3 and trial["wrong"]*100<=trial["admitted"]:
            candidates.append((trial["correct"],gate))
    best=max(candidates,default=(0,(1.01,1.01,1.01)))[1]
    return dict(gate=best,calibration=run_policy(data["dev"],score["dev"],best),
                held=run_policy(data["test"],score["test"],best,True),
                raw=run_policy(data["test"],score["test"],(0.,0.,0.),True))


if __name__=="__main__":
    destination=ROOT.parent/"assertions.json"
    destination.write_text(json.dumps(main(),indent=2))
