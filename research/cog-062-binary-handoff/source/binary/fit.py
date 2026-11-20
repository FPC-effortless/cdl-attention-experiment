"""Train A/B only. No E labels, E model prediction, official dev/test, or tuning."""
import json,hashlib,os,sys,time
from collections import Counter
from pathlib import Path
if sys.flags.optimize: raise RuntimeError("optimized Python not supported")
import numpy as np
import joblib
from threadpoolctl import threadpool_limits, threadpool_info
from model import WIDTH,base_features,meta_features,meta_text,classifier,shortlist,rank_targets,hit_supervision
from trainlab.research_adequacy import recipe_identity
ROOT=Path(__file__).resolve().parent
def write(name,doc): (Path(os.environ["TRAINLAB_OUTPUT_DIR"])/name).write_text(json.dumps(doc,allow_nan=False,sort_keys=True))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_fit_paths(paths):
    banned={'E.json','E-input.json','E-gold.json','E-overlap.json'}
    if any(Path(p).name in banned for p in paths):
        raise ValueError("outcome sealed out of training inputs")

def main():
    cfg=json.loads((ROOT/"recipe.json").read_bytes());ptr=cfg["research_adequacy"]
    contract=json.loads((ROOT/ptr["file"]).read_bytes());pins={p:sha(ROOT/p) for p in cfg["inputs"]}
    if pins[ptr["file"]]!=ptr["sha256"] or contract["inputs"]!={p:h for p,h in pins.items() if p!=ptr["file"]}: raise ValueError("fit input closure changed")
    a,b=(json.loads((ROOT/"data"/(p+".json")).read_bytes()) for p in "AB")
    if {r["group"] for r in a}&{r["group"] for r in b}: raise ValueError("training groups overlap")
    validate_fit_paths(pins)
    obs={};report={};start=time.perf_counter();model={};base=[]
    with threadpool_limits(limits=1):
        threads=threadpool_info()
        if not threads or any(x["num_threads"]!=1 for x in threads): raise ValueError("expected one numerical worker")
        ax,vec=base_features(a);bx,_=base_features(b,vec)
        model["vectorizers"]=vec
        targets=[]
        for j,name in enumerate(("area","price")):
            ix=[i for i,r in enumerate(a) if r["delta"][j] is not None]
            m,conv=classifier(ax[ix],[a[i]["delta"][j] for i in ix],250,8)
            base.append(m)
            ids,p=shortlist(m.predict_proba(bx));labels=np.array([r["delta"][j] for r in b],dtype=object)
            usable=np.array([y is not None for y in labels])
            y=rank_targets(labels[usable],m.classes_[ids[usable]])
            nr=Counter(y.tolist());report[name+"_rank_rows"]={str(k):v for k,v in nr.items()};report[name+"_base_iterations"]=int(m.n_iter_.max());report[name+"_base_converged"]=conv
            obs[name+"_rank2"]=nr[1];obs[name+"_neither"]=nr[2];obs[name+"_base_iterations"]=int(m.n_iter_.max());obs[name+"_base_parameters"]=int(m.coef_.size+m.intercept_.size)
            group_counts={k:len({b[int(i)]["group"] for i in np.flatnonzero(usable)[y==k]}) for k in (0,1,2)}
            report[name+"_rank_groups"]={str(k):v for k,v in group_counts.items()}
            mask,hit_targets=hit_supervision(usable,y)
            if int(mask.sum())!=nr[0]+nr[1]:raise ValueError("B candidate-hit mask mismatch")
            report[name+"_binary_supervised_rows"]=int(mask.sum())
            report[name+"_binary_excluded_misses"]=nr[2]
            obs[name+"_rank1"]=nr[0]
            obs[name+"_rank1_groups"]=group_counts[0];obs[name+"_rank2_groups"]=group_counts[1]
            write("fit_counts.json",report)
            if any(nr[k]<100 or group_counts[k]<50 for k in (0,1)):
                raise ValueError("insufficient B candidate-hit target classes before E access")
            targets.append((ids,p,mask,hit_targets))
        model["base"]=base
        for mode in ("canonical","ordered"):
            text=meta_text(b,mode);model[mode]=[]
            for name,(ids,p,mask,y) in zip(("area","price"),targets):
                x=meta_features(text,ids,p);m,conv=classifier(x[mask],y,200,1)
                if m.classes_.tolist()!=[0,1] or m.coef_.shape!=(1,WIDTH):raise ValueError("wrong fixed re-ranker size")
                model[mode].append(m)
                tag=name+"_"+mode
                report[tag+"_iterations"]=int(m.n_iter_.max());report[tag+"_converged"]=conv
                obs[tag+"_params"]=int(m.coef_.size+m.intercept_.size);obs[tag+"_rows"]=int(mask.sum());obs[tag+"_iterations"]=int(m.n_iter_.max())
        report["fit_seconds"]=time.perf_counter()-start
    if any(report[k] is False for k in report if k.endswith("_converged")): report["optimization_adequacy"]="CAP_NOT_CONVERGED_REPORT_ONLY"
    else: report["optimization_adequacy"]="SOLVERS_CONVERGED"
    dimensions=["area","price"]
    obs.update(eligible_a_turns=len(a),eligible_b_turns=len(b),fit_calls=6,one_thread=1,all_converged=int(report["optimization_adequacy"]=="SOLVERS_CONVERGED"),same_meta_rows=int(all(obs[k+"_ordered_rows"]==obs[k+"_canonical_rows"] for k in dimensions)),official_dev_test_reads=0,eval_in_fit=0)
    write("fit_counts.json",report)
    joblib.dump(model,Path(os.environ["TRAINLAB_OUTPUT_DIR"])/"heads.joblib",compress=3)
    write("metrics.json",{"fit_calls":6})
    write("observations.json",dict(schema="trainlab.execution-adequacy-observations.v1",recipe_sha256=recipe_identity(cfg),inputs=pins,producer={"file":"fit.py","sha256":pins["fit.py"]},observations=obs))
if __name__=="__main__": main()
