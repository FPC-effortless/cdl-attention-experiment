"""Weight-free, post-prediction family accounting. No model dependencies."""
import argparse
from collections import Counter, defaultdict
import json
import math
from .data import ANSWERS


def unique(rows):
    joined={}
    for r in rows:
        k=(r["family_id"],r["world_index"])
        if k in joined or k[1] not in (0,1):
            raise ValueError("duplicated family orientation")
        joined[k]=r
    return joined


def score(predictions,truth,answer_ids):
    if set(answer_ids)!=set(ANSWERS) or len(set(answer_ids.values()))!=len(ANSWERS):
        raise ValueError("invalid answer token identity")
    p,t=unique(predictions),unique(truth)
    if not t or set(p)!=set(t):
        raise ValueError("prediction and truth joins are not exact")
    groups=defaultdict(Counter)
    for f in sorted({f for f,_ in t}):
        rr=[p[f,i] for i in (0,1)]
        tt=[t[f,i] for i in (0,1)]
        if tt[0]["answer"]==tt[1]["answer"] or tt[0]["delay"]!=tt[1]["delay"]:
            raise ValueError("invalid pair truth")
        passes,ablated,positive_margin,hit,hits_color=[],[],[],[],[]
        for i,(r,k) in enumerate(zip(rr,tt)):
            answer,foil=k["answer"],tt[1-i]["answer"]
            right,wrong=ANSWERS.index(answer),ANSWERS.index(foil)
            sv=r["answer_scores"]
            if len(sv)!=len(ANSWERS) or not all(math.isfinite(v) for v in sv):
                raise ValueError("invalid frozen scores")
            passes.append(r["prediction"]==answer_ids[answer])
            positive_margin.append(sv[right]>sv[wrong])
            if "no_route_prediction" in r:
                ablated.append(r["no_route_prediction"]==answer_ids[answer])
            here=r["prompt_length"]-1
            left,right=k["support_start_token"],k["support_end_token"]
            if not 0<=left<=k["answer_source_token"]<=right<here:
                raise ValueError("support metadata outside visible prefix")
            cand=[z for pipe in r["route_positions"].values() for z in pipe]
            if any(len(pipe)!=4 for pipe in r["route_positions"].values()) or any(
                    type(z) is not int or z < -1 or z>=here for z in cand):
                raise ValueError("impossible routing position")
            hit.append(any(left<=z<=right for z in cand))
            hits_color.append(k["answer_source_token"] in cand)
        eligible=all(k["age_eligible"] for k in tt)
        same=all(k["local_suffix_equal"] for k in tt)
        strata=["all",f"delay_{tt[0]['delay']}","both_age_eligible" if eligible else "age_limited",
                "same_local_suffix" if same else "different_local_suffix"]
        for g in strata:
            c=groups[g]
            c["families"]+=1
            c["both_exact"]+=all(passes)
            c["world_exact"]+=sum(passes)
            c["both_gold_over_foil"]+=all(positive_margin)
            c["both_literal_support_hits"]+=all(hit)
            c["both_color_position_hits"]+=all(hits_color)
            if ablated:
                if len(ablated)!=2:raise ValueError("partial intervention pair")
                c["ablated_both_exact"]+=all(ablated)
                c["ordinary_success_lost_when_disabled"]+=all(passes) and not all(ablated)
                c["ordinary_failure_helped_when_disabled"]+=not all(passes) and all(ablated)
    return {g:dict(c) for g,c in groups.items()}


def main():
    import hashlib
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("predictions","truth","vocabulary","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    for name in ("predictions","truth","vocabulary"):
        parser.add_argument("--"+name+"-sha256",required=True)
    a=parser.parse_args()
    def verified(key):
        b=getattr(a,key).read_bytes()
        if hashlib.sha256(b).hexdigest()!=getattr(a,key+"_sha256"):
            raise ValueError("score input digest mismatch")
        return b
    pred=[json.loads(x) for x in verified("predictions").splitlines()]
    truth=[json.loads(x) for x in verified("truth").splitlines()]
    v=json.loads(verified("vocabulary"))
    ids={w:len(v["special"])+256+i for i,w in enumerate(v["words"])}
    result=score(pred,truth,{k:ids[k] for k in ANSWERS})
    with a.output.open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
