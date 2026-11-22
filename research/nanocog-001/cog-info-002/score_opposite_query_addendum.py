"""INFO002 secondary analysis: exact raw-prompt pair joins, no model code."""
from collections import Counter
from hashlib import sha256
from math import isfinite
from pathlib import Path
from struct import Struct
import json

from trainlab.runner import verify_run

R = Path(__file__).resolve().parent
PINS = {
    "INFO002_SEAL.json": "538f1abd5cdea45bc0b8a3912a1ed2df071d132a2e8f14d33bcc755db71aac7e",
    "panel/actor.json": "41e1c19a805ac0061252113303d4bf4e5b0e2a3a3c582df4ba67f5763d858761",
    "panel/judge.json": "3c403237f504fc4a40baa1fea144e00263a1d81ef67c03c6a99eee0e1c9a38f1",
    "info002_admissions/ledger.json": "425737b22fb1916dacedaba9aa630c779cb9be193e03844849ab352f4f208304",
    "INFO002_RESULTS.json": "b1e31411f1661bf708a722974b9d305964d9d89993805b26e90a5965bb028fe8",
}
BRANCH = ("base", "edit_first_native", "edit_second_native", "edit_first_gate", "edit_first_taps",
          "edit_first_joint", "edit_second_gate", "edit_second_taps", "edit_second_joint")
ARMS = ("native", "taps")
PAIRMET = ("target", "preserved", "pair_complete", "distractor_insert", "distractor_insert_and_target",
           "base_preserved", "base_to_edit_preserved_harm", "base_to_edit_preserved_restoration",
           "base_to_edit_harm_and_target", "target_only", "preserved_only", "neither")
CROSS = ("target", "preserved", "pair_complete", "distractor_insert")
BOTH = ("both_pairs_complete", "first_only_complete", "second_only_complete", "neither_pair_complete",
        "both_targets_correct", "both_preserved_correct", "either_distractor_insert", "both_distractor_insert",
        "first_pair_complete_second_wrong_insert", "second_pair_complete_first_wrong_insert")

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def digest(b):
    return sha256(b).hexdigest()

def read(path, pin, raw=False):
    b = (R/path).read_bytes()
    require(digest(b) == pin, f"checksum mismatch: {path}")
    return b if raw else json.loads(b)

def build_histories(actor, judge):
    out, raw_ids, group_keys = [], set(), set()
    variants = ("base", "edit_first", "edit_second")
    require(len(actor["groups"]) == len(judge["groups"]) == 32, "group count")
    for g, (ag, jg) in enumerate(zip(actor["groups"], judge["groups"], strict=True)):
        partial = {}
        names = tuple(jg["name_pair"])
        require(jg["group"] == g and len(ag["base"]) == len(jg["cells"]) == 8 and names not in group_keys, "group shape")
        group_keys.add(names)
        for i, rs in enumerate(zip(*(ag[k] for k in variants), strict=True)):
            require(not raw_ids.intersection(rs) and len(set(rs))==3, "raw collisions")
            raw_ids.update(rs)
            base, *edited = [tuple(actor["prompts"][r].split()) for r in rs]
            n, link1, x, m, link2, y, q = base
            require({n,m}==set(names) and q in (n,m) and link1==link2=="is", "base syntax")
            insert = edited[0][2]
            require(len({x,y,insert})==3 and set((x,y,insert))==set(jg["color_triple"]) and insert==edited[1][5], "color semantics")
            toks = x,y
            for e, loc in zip(edited,(2,5),strict=True):
                require([k for k,(a,b) in enumerate(zip(base,e,strict=True)) if a!=b]==[loc], "physical edit")
            cell = jg["cells"][i]
            require(cell["rows"]==dict(zip(variants,rs,strict=True)) and
                    cell["query_name"]==q and cell["base_answer"]==toks[(n,m).index(q)]
                    and cell["inserted_color"]==insert, "raw/judge identity")
            same = partial.setdefault(base[:-1],{})
            require(q not in same,"duplicate query")
            same[q] = (8*g+i,base,edited,insert)
        require(len(partial)==4, "four group histories")
        require({(h[0],h[2],h[3],h[5]) for h in partial} ==
                {(a,u,b,v) for a,b in (names,names[::-1])
                 for u,v in (tuple(jg["baseline_colors"]),tuple(reversed(jg["baseline_colors"])))}, "complete assignment × order")
        for h, rec in sorted(partial.items()):
            n,_,x,m,_,y=h
            require(set(rec)=={n,m}, "opposite queries")
            left,right=rec[n],rec[m]
            require(left[1][:-1]==right[1][:-1] and left[3]==right[3] and
                    [e[:-1] for e in left[2]]==[e[:-1] for e in right[2]], "exact history across queries")
            out.append((left[0],right[0],x,y,left[3]))
    require(raw_ids==set(range(768)) and len(out)==128, "global scope")
    return out

def arm_outcome(z, words, hist, edit, arm):
    l,r,x,y,c=hist
    target_row,dist_row,old = (l,r,y) if edit=="first" else (r,l,x)
    def strict(branch, row, ans):
        v=z[branch][row]
        return v[words[ans]]==max(v) and v.count(v[words[ans]])==1
    b="edit_"+edit+"_"+arm
    a=strict(b,target_row,c); p=strict(b,dist_row,old)
    d=strict(b,dist_row,c); bp=strict("base",dist_row,old)
    return dict(zip(PAIRMET,(a,p,a and p,d,d and a,bp,bp and not p,p and not bp,
                              bp and not p and a,a and not p,p and not a,not a and not p),strict=True))

def cross(a,b,keys):
    return {k:{"native_only":sum(x[k] and not y[k] for x,y in zip(a,b,strict=True)),
               "taps_only":sum(y[k] and not x[k] for x,y in zip(a,b,strict=True)),
               "both":sum(y[k] and x[k] for x,y in zip(a,b,strict=True)),
               "neither":sum(not y[k] and not x[k] for x,y in zip(a,b,strict=True))}
            for k in keys}

def counts(x, keys):
    return {k:sum(q[k] for q in x) for k in keys}

def both(a,b):
    x=a["pair_complete"]; y=b["pair_complete"]
    d=a["distractor_insert"]; e=b["distractor_insert"]
    return dict(zip(BOTH,(x and y,x and not y,y and not x,not x and not y,
        a["target"] and b["target"],a["preserved"] and b["preserved"],d or e,d and e,
        x and e,y and d),strict=True))

def analyze(fixed):
    seal,actor,judge,ledger=(fixed[k] for k in list(PINS)[:4])
    require(set(seal["runs"])==set(ledger)=={"info002-1171","info002-1371"} and
            seal["ledger_sha256"]==PINS["info002_admissions/ledger.json"], "seal scope")
    h=build_histories(actor,judge)
    out={}
    for name,s in sorted(seal["runs"].items()):
        p=Path("info002_runs")/s["run_id"]; o=p/"output"
        run=read(p/"run.json",s["run_sha256"]); a=s["artifacts"]
        require(verify_run(R/p)["valid"] and run["status"]=="succeeded" and run["exit_code"]==0 and
                run["inputs"]==ledger[name]["inputs"] and run["artifacts"]==a,"native receipt")
        m=read(o/"intervention_receipt.json",a["intervention_receipt.json"])
        require(m["actor_sha256"]==PINS["panel/actor.json"] and m["shape"]==[9,256,287] and
                tuple(m["branch_order"])==BRANCH and m["logits_sha256"]==a["LOGITS.f32le"] and
                m["checkpoint_sha256"]==ledger[name]["checkpoint_sha256"] and
                m["seed"]==ledger[name]["seed"],"wire provenance")
        words={t:262+i for i,t in enumerate(m["tokenizer_words"])}
        require(len(words)==25 and set(words.values())==set(range(262,287)),"word IDs")
        wire=list(Struct("<287f").iter_unpack(read(o/"LOGITS.f32le",a["LOGITS.f32le"],True)))
        require(len(wire)==9*256 and all(isfinite(v) for row in wire for v in row), "finite wire")
        z={k:wire[j*256:(j+1)*256] for j,k in enumerate(BRANCH)}
        per={e:{k:[arm_outcome(z,words,i,e,k) for i in h] for k in ARMS} for e in ("first","second")}
        per["all"]={k:per["first"][k]+per["second"][k] for k in ARMS}
        agg={e:dict(n=len(w["native"]),arms={k:counts(w[k],PAIRMET) for k in ARMS},
                    paired_tap_vs_native=cross(w["native"],w["taps"],CROSS)) for e,w in per.items()}
        cw={k:[both(a,b) for a,b in zip(per["first"][k],per["second"][k],strict=True)] for k in ARMS}
        hist=dict(n=len(h),arms={k:counts(cw[k],BOTH) for k in ARMS},
                  paired_tap_vs_native=cross(cw["native"],cw["taps"],("both_pairs_complete","both_targets_correct",
                        "both_preserved_correct","either_distractor_insert")))
        off=fixed["INFO002_RESULTS.json"]["results"][name]
        for k in ARMS:
            for e in ("first","second"):
                opposite="second" if e=="first" else "first"
                expected_t=off[e]["summary"][k]; expected_d=off[opposite]["summary"][k]
                observed=agg[e]["arms"][k]
                require((observed["target"],observed["preserved"],observed["distractor_insert"],
                         observed["base_to_edit_preserved_harm"],observed["base_to_edit_preserved_restoration"])==
                        (expected_t["target_correct_donor_gold"],expected_d["distractor_correct_original_gold"],
                         expected_d["distractor_inappropriately_selects_insert"],expected_d["distractor_harms"],
                         expected_d["distractor_restorations"]), "original marginals failed independent reconciliation")
        out[name]=dict(run_id=s["run_id"],run_sha256=s["run_sha256"],artifact_hashes=a,
                       checkpoint_sha256=m["checkpoint_sha256"],same_physical_edit=agg,matched_history_four=hist)
    return out

if __name__=="__main__":
    fixed={k:read(k,v) for k,v in PINS.items()}
    lock=R/"INFO002_OPPOSITE_QUERY_ADDENDUM_LOCK.md"
    locks=json.loads((R/"OPPOSITE_QUERY_SCORE_LOCK.json").read_bytes())
    require(locks=={p.name:digest(p.read_bytes()) for p in (lock,Path(__file__))}, "predecode source lock")
    out=R/"INFO002_OPPOSITE_QUERY_ADDENDUM.json"
    result=dict(status="SECONDARY_POST_HOC_REUSED_INFO002_ONLY",input_sha256=PINS,source_sha256=locks,
                predecode_lock_sha256=digest((R/"OPPOSITE_QUERY_SCORE_LOCK.json").read_bytes()),
                model_invocations=0,optimizer_updates=0,results=analyze(fixed))
    with out.open("x") as f:
        json.dump(result,f,sort_keys=True,allow_nan=False,indent=2);f.write("\n")
    print(out,digest(out.read_bytes()))
