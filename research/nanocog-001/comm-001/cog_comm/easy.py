"""EASY calibration blind predictor and sealed scorer (separate role invocations)."""
import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys
import time

# The score-only role can stage this single module, using standard library.
# Model and checkpoint loader are imported only inside the prediction role.
def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    with path.open("rb") as f:return file_digest(f,"sha256").hexdigest()


def strict_json(text):
    def obj(items):
        keys=dict(items)
        require(len(keys)==len(items),"duplicate JSON key")
        return keys
    def nonfinite(x):
        raise ValueError("invalid JSON numeric term "+x)
    return json.loads(text,object_pairs_hook=obj,parse_constant=nonfinite)


def write(path,value):
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

SEEDS=(171,271,371)
N_NEAR=120
N_COLLISION=120
N_WORLDS=2*(N_NEAR+N_COLLISION)


def read_actor(path):
    rows=[strict_json(x) for x in path.read_text(encoding="utf8").splitlines()]
    require(len(rows)==N_WORLDS and
            all(type(r) is dict and set(r)=={"prompt"} and type(r["prompt"]) is str and r["prompt"].strip()
                for r in rows),"only exact easy actor prompts permitted")
    ps=[r["prompt"] for r in rows]
    require(len(set(ps))==len(ps),"duplicate EASY actor prompt")
    return ps


def write_jsonlines(path,rows):
    with path.open("x",encoding="utf8") as fd:
        for row in rows:fd.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")


def predict(actor, actor_sha256, checkpoint, checkpoint_sha256, seed, out):
    import torch
    from .fit import Plan
    from .model import build_decoder
    from .objective import Tokenizer
    started=time.perf_counter()
    require(sys.version_info[:2]==(3,12) and torch.__version__=="2.10.0+cpu","bad frozen CPU runtime")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1,"thread cap refused")
    require(digest(actor)==actor_sha256 and digest(checkpoint)==checkpoint_sha256,"exact EASY prediction digest mismatch")
    prompts=read_actor(actor)
    ck=torch.load(checkpoint,weights_only=True,map_location="cpu")
    require(type(ck) is dict and set(ck)=={"format","plan","tokenizer_words","vocab_size","train_sha256",
            "vocab_sha256","schedule_sha256","updates","state_dict"},"unknown FIT checkpoint schema")
    plan=Plan("A","learned",seed)
    require(ck["format"]==1 and ck["plan"]==asdict(plan) and ck["updates"]==plan.steps,
            "EASY requires exact fixed A fit")
    tok=Tokenizer(tuple(ck["tokenizer_words"]))
    require(tok.size==ck["vocab_size"] and all(type(w) is str for w in tok.words),"tokenizer fit changed")
    model=build_decoder(tok.size,"A",width=32,layers=2,block=8,retained_blocks=64,seed=seed).eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    encoded=[tok.encode(p) for p in prompts]
    require(all(8<=len(x)<=512 for x in encoded) and all(len(l)==len(r) for l,r in zip(encoded[::2],encoded[1::2])),
            "EASY actor batch geometry differs")
    seen=0; sink=[]
    with torch.inference_mode():
        for i in range(0,len(encoded),8):
            block=encoded[i:i+8]
            lengths=torch.tensor([len(x) for x in block])
            xx=torch.tensor([r+[0]*(int(lengths.max())-len(r)) for r in block],dtype=torch.long)
            logits,_=model(xx,state=None)
            tail=logits[torch.arange(len(block)),lengths-1]
            require(tail.shape==(len(block),tok.size) and bool(torch.isfinite(tail).all()),"invalid EASY logits")
            for row in tail.tolist():
                sink.append({"logits":row})
            seen+=len(block)
    out.mkdir(parents=True,exist_ok=True)
    write_jsonlines(out/"predictions.jsonl",iter([
      {"seed":seed,"actor_sha256":actor_sha256,"checkpoint_sha256":checkpoint_sha256,"fit_train_sha256":ck["train_sha256"],
       "vocab_sha256":ck["vocab_sha256"],"vocab_size":tok.size,"scope":"easy-cal-blind","worlds":seen},
       *sink]))
    counts={"eval_worlds":seen,"eval_pairs":seen//2,"model_initializations":1,
            "input_real_tokens":sum(map(len,encoded)),"max_input_tokens":max(map(len,encoded)),
            "optimizer_steps":0,"main_study_inputs":0,"label_inputs":0}
    write(out/"native-counters.json",{"observations":counts,"metrics":{"worlds":seen}})
    write(out/"predict.json",{"elapsed_seconds":time.perf_counter()-started,**counts})


def _near_pass(correct, paired, per_color, collision_pairs, collision_individual, collisions_same):
    return (correct>=228 and paired>=108 and bool(per_color) and
            len(per_color)==8 and all(n==30 and c>=27 for c,n in per_color.values()) and
            collision_pairs==0 and collision_individual<=120 and collisions_same==120)


def verify(truth, truth_sha256, prediction, prediction_sha256, out):
    require(digest(truth)==truth_sha256 and len(prediction)==len(prediction_sha256)==len(SEEDS),
            "wrong sealed EASY inputs or run roster")
    gold=strict_json(truth.read_text())
    require(type(gold) is dict and set(gold)=={"scope","actor_sha256","vocab_sha256","vocab_size","fit_train_sha256","pairs"}
            and gold["scope"]=="easy-cal-only" and type(gold["vocab_size"]) is int,
            "wrong scoped truth schema")
    vocab=gold["vocab_size"]
    pairs=gold["pairs"]
    require(type(pairs) is list and len(pairs)==N_NEAR+N_COLLISION,"wrong calibration roster")
    from collections import Counter
    for r in pairs:
        require(type(r) is dict and set(r)=={"kind","answers"} and r["kind"] in ("near","collision")
            and type(r["answers"]) is list and len(r["answers"])==2 and
            all(type(a) is int and 0<=a<vocab for a in r["answers"])
            and r["answers"][0]!=r["answers"][1],"wrong calibration pair coordinates")
    require(Counter(r["kind"] for r in pairs)=={"near":N_NEAR,"collision":N_COLLISION},"wrong calibration pair split")
    labels={k:Counter(a for r in pairs if r["kind"]==k for a in r["answers"]) for k in ("near","collision")}
    all_answers={a for r in pairs for a in r["answers"]}
    require(len(all_answers)==8 and all(set(c)==all_answers and set(c.values())=={30} for c in labels.values()),
            "EASY color support is not balanced")
    scores=[]; seen_ck=set()
    for seed,path,h in zip(SEEDS,prediction,prediction_sha256):
        require(digest(path)==h,"blind prediction digest differs")
        rows=[strict_json(l) for l in path.read_text().splitlines()]
        require(len(rows)==1+N_WORLDS,"incomplete blind worlds")
        hdr,*rows=rows
        expected={k:gold[k] for k in ("actor_sha256","fit_train_sha256","vocab_sha256","vocab_size")}
        require(type(hdr) is dict and set(hdr)==set(expected)|{"seed","checkpoint_sha256","scope","worlds"}
            and hdr["scope"]=="easy-cal-blind" and hdr["worlds"]==N_WORLDS and hdr["seed"]==seed
            and all(hdr[k]==v for k,v in expected.items())
            and isinstance(hdr["checkpoint_sha256"],str) and len(hdr["checkpoint_sha256"])==64
            and hdr["checkpoint_sha256"] not in seen_ck,"prediction closure/fit lineage drift")
        seen_ck.add(hdr["checkpoint_sha256"])
        for row in rows:
            require(type(row) is dict and set(row)=={"logits"} and type(row["logits"]) is list
                and len(row["logits"])==vocab and
                all(type(s) in (int,float) and math.isfinite(s) for s in row["logits"]),"invalid full-vocabulary logits")
        near_world=near_paired=collision_world=collision_paired=collision_same=near_ties=collision_ties=0
        near_by_color={a:[0,n] for a,n in labels["near"].items()}
        for idx,rec in enumerate(pairs):
            a,b=rows[2*idx]["logits"],rows[2*idx+1]["logits"]
            top_a=max(a);top_b=max(b)
            ties=(a.count(top_a)!=1,b.count(top_b)!=1)
            true=rec["answers"]
            hit=((not ties[0]) and a.index(top_a)==true[0],(not ties[1]) and b.index(top_b)==true[1])
            if rec["kind"]=="near":
                near_world+=sum(hit);near_paired+=all(hit);near_ties+=sum(ties)
                for answer,ok in zip(true,hit):near_by_color[answer][0]+=ok
            else:
                collision_world+=sum(hit);collision_paired+=all(hit);collision_ties+=sum(ties)
                collision_same+=a==b
        scores.append({"seed":seed,"near_world_correct":near_world,"near_world_total":2*N_NEAR,
            "near_complete_pairs":near_paired,"near_pairs_total":N_NEAR,"near_per_color":near_by_color,
            "near_world_ties":near_ties,"collision_world_correct":collision_world,
            "collision_complete_pairs":collision_paired,"collision_exact_logits_pairs":collision_same,
            "collision_world_ties":collision_ties,"collision_pairs_total":N_COLLISION,
            "passed":_near_pass(near_world,near_paired,near_by_color,collision_paired,collision_world,collision_same)})
    out.mkdir(parents=True,exist_ok=True)
    passed=all(r["passed"] for r in scores)
    write(out/"score.json",{"scope":"constructed EASY pipeline sensitivity only",
          "passed_all_seeds":passed,"pass_thresholds":{"near_world":228,"near_pairs":108,
            "near_min_per_color_out_of_30":27,"collision_exact_logits_pairs":120,
            "collision_world_max":120,"collision_complete_pairs":0},
          "truth_sha256":truth_sha256,"predictions_sha256":list(prediction_sha256),"scores":scores})
    counts={"scored_seed_count":len(scores),"scored_near_worlds":len(scores)*2*N_NEAR,
            "scored_collision_worlds":len(scores)*2*N_COLLISION,"model_initializations":0,
            "optimizer_steps":0,"main_study_inputs":0,"weight_inputs":0}
    write(out/"native-counters.json",{"observations":counts,"metrics":{"passed_seeds":sum(r["passed"] for r in scores)}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="phase",required=True)
    pp=sub.add_parser("predict")
    pp.add_argument("--seed",required=True,type=int,choices=SEEDS)
    pv=sub.add_parser("verify")
    for parser,files in ((pp,("actor","checkpoint")),(pv,("truth","prediction"))):
        parser.add_argument("--out",required=True,type=Path)
        for name in files:
            kws={"action":"append"} if name=="prediction" else {}
            parser.add_argument("--"+name,required=True,type=Path,**kws)
            parser.add_argument("--"+name+"-sha256",required=True,**kws)
    a=vars(p.parse_args());phase=a.pop("phase")
    {"predict":predict,"verify":verify}[phase](**a)


if __name__=="__main__":
    main()
