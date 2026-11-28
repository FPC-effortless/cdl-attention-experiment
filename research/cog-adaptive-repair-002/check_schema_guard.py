"""Regression checks for authenticated learned roles, abstention and stale state."""
import argparse
import json
import sys
from pathlib import Path

import torch

from schema_guard import GuardedStore, build_adapter, guarded_actions, validate
from structured_memory import TRUST, parse_prompts
from structured_transfer import load_tagger


def check(seed, tagger, base):
    from confirm_repair import _episodes, _counts, PANELS, SEED
    known = b"The log lists Ada at 2. "
    fresh = b"The log lists Ada at 3. "
    weak = b"A rumor puts Ada at 1. "
    unknown = b"The mailbox alleges Ada received 1. "
    query = b"What digit does the log give for Ada?"
    cases = {
        "authorized": (known+query, ord("2")),
        "known_weak_hold": (known+weak+query, ord("2")),
        "unknown_authority": (unknown+query,None),
        "missing_subject": (known+query.replace(b"Ada",b"Ben"),None),
        "unsupported_query": (known+b"Speak the number of Ada!",None),
        "stale": (known+unknown+query,None),
        "rebuilt": (known+unknown+fresh+query,ord("3")),
        "overflow": (known+b"X"*180+b". "+query,None),
        "outside_digit_domain": (known+fresh.replace(b"3",b"9")+query,None),
        "punctuation": (known+fresh.replace(b"Ada",b"Ada_")+query,None),
        "fresh_name": (known.replace(b"Ada",b"Kemi")+query.replace(b"Ada",b"Kemi"),ord("2")),
    }
    outcomes = {}
    for label,(raw,expected) in cases.items():
        s = GuardedStore(tagger)
        direct = {i:p for i,b in enumerate(raw) if (p:=s.step(b)) is not None}
        cache = guarded_actions(raw,parse_prompts(tagger,[raw]))
        assert cache == direct and direct.get(len(raw)-1) == expected,(seed,label,direct)
        outcomes[label]=expected
    assert not validate(unknown,(TRUST,b"ada",ord("1"),0))
    assert not validate(known,(TRUST,b"ben",ord("2"),0))
    assert not validate(known,(TRUST,b"ada",ord("3"),0))
    # Fork a previously accepted stream state and invalidate only one branch.
    g=build_adapter(base,tagger)
    def ids(t):return torch.tensor([list(t)])
    _,parent = g(ids(known),carry=True)
    restored,st = g(ids(unknown+query),parent,True)
    accepted,_=g(ids(query),parent,True)
    standalone=g(ids(known+unknown+query))
    error=(standalone[:,-len(unknown+query):]-restored).abs().max().item()
    assert error < 3e-5 and int(accepted[0,-1].argmax())==ord("2")
    assert len(st[1][0].entries)==0
    panels = {}
    for name in PANELS:
        episodes=_episodes(name,64,SEED)
        prompts=[x for e in episodes for x in e[:2]]
        parsed=parse_prompts(tagger,prompts)
        choices=[guarded_actions(t,parsed).get(len(t)-1,-1) for t in prompts]
        panels[name]=_counts(episodes,choices)
        panels[name]["dispatched"]=sum(p!=-1 for p in choices)
    return {"seed":seed,"cases":outcomes,"branch_and_stream_max_error":error,"panels":panels}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--taggers",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"unified-search-002"))
    from integrated import WholeCOG
    base=WholeCOG("SS_native_native",9731).eval()
    base.load_state_dict(torch.load(args.parent,weights_only=True))
    results=[]
    for seed in (9731,9743,9769):
        results.append(check(seed,load_tagger(args.taggers/f"tagger_{seed}.pt"),base))
    args.out.write_text(json.dumps(results,indent=2)+"\n")
    print([(r['seed'],r['branch_and_stream_max_error'],{n:v['both_correct'] for n,v in r['panels'].items()}) for r in results])
