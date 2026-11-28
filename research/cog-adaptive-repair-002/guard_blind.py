"""Post-hoc check of the narrow schema restriction on previously opened prompts."""
import json
import sys
from pathlib import Path
import torch
from blind_transfer import evaluate_solver
from schema_guard import guarded_actions
from structured_memory import parse_many, segments
from structured_transfer import load_tagger

if __name__=="__main__":
    torch.set_num_threads(1)
    raw=[]
    evaluate_solver(lambda p: raw.append(p))
    clauses=sorted({c for p in raw for _,c in segments(p)})
    results={}
    for path in sorted(Path(sys.argv[1]).glob("tagger_*.pt")):
        parsed=parse_many(load_tagger(path),clauses)
        by_prompt={p:guarded_actions(p,parsed).get(len(p)-1) for p in raw}
        score=evaluate_solver(by_prompt.__getitem__)
        results[path.name]=score["overall"]
    Path(sys.argv[2]).write_text(json.dumps(results,indent=2))
