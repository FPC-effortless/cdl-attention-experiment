"""Train all strict and simultaneous-supervision comparisons on the archived split."""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from models import cues, fit, history, prob, train, txt_features
from evaluation import eligible, measure_admissions, summarize, threshold_selection

ROOT = Path(__file__).resolve().parent


def run(seed, out):
    rows = json.loads((ROOT / "combined.json").read_text())
    sem = np.load(ROOT / "combined.npy")
    y = np.array([r["target"] for r in rows])
    strict = np.array([r["strict"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    group = np.array([r["issue"] for r in rows])
    tr = np.flatnonzero((sp == "train") & (strict | np.array([r["delta_seconds"] == 0 and r["target"] for r in rows],bool)))
    va = np.flatnonzero((sp == "validation") & strict)
    texts = np.array([r["text"][:12000] for r in rows],dtype=object)
    cue = np.array([cues(r) for r in rows],float)
    ctx = history(rows)
    heads, fitted = {}, {}
    def experts(ids, evaluate, save):
        xx, vector = txt_features(texts[np.r_[ids,evaluate]],np.arange(len(ids)))
        labels = y[ids]
        tx, ex = xx[:len(ids)], xx[len(ids):]
        all_predictions = []
        for arm, mask in (("strict",strict[ids]),("pooled",np.ones(len(ids),bool))):
            # Select expert regularization on the archived validation split in the full fit.
            if save:
                both = np.r_[ids,evaluate]
                val_local = np.flatnonzero((sp[both]=="validation") & strict[both])
                loc = np.flatnonzero(mask)
                lexical = fit(xx,y[both],loc,val_local)
                prior = fit(ctx[both],y[both],loc,val_local)
                fitted[arm] = (lexical,prior)
                flat_x = hstack([xx,csr_matrix(np.c_[ctx[both],cue[both],sem[both]])],format="csr")
                flat = fit(flat_x,y[both],loc,val_local)
                fitted[arm+"_flat"] = flat
                pred = np.empty(len(rows))
                pred[both] = prob(flat,flat_x)
                heads[arm+"_flat"] = pred
            else:
                lex_c, prior_c = [m.C for m in fitted[arm]]
                lexical = train(tx[mask],labels[mask],lex_c)
                prior = train(ctx[ids][mask],labels[mask],prior_c)
            all_predictions += [prob(lexical,ex),prob(prior,ctx[evaluate])]
        auxmask = ~strict[ids] | (labels==0)
        aux = train(tx[auxmask],(~strict[ids][auxmask]).astype(int),3)
        all_predictions.append(prob(aux,ex))
        if save:
            fitted.update(aux=aux,vectors=vector)
        return np.stack(all_predictions,axis=1)
    others = np.setdiff1d(np.arange(len(rows)),tr)
    base = np.zeros((len(rows),5))
    base[others] = experts(tr,others,True)
    for a,b in GroupKFold(4,shuffle=True,random_state=seed).split(tr,groups=group[tr]):
        base[tr[b]] = experts(tr[a],tr[b],False)
    for name, cols, ids in (("strict_modular",[0,1],tr[strict[tr]]),("pooled_modular",[2,3],tr),
                             ("separate_expert",[0,1,4],tr[strict[tr]])):
        x = np.c_[logit(np.clip(base[:,cols],1e-5,1-1e-5)),ctx[:,0],cue,sem]
        m = fit(x,y,ids,va)
        heads[name] = prob(m,x)
        fitted[name] = m
    joblib.dump(fitted,out/"issue-checkpoint.joblib",compress=3)
    metrics, predictions = {}, []
    for name,p in heads.items():
        t,k = threshold_selection(p,cue,sem,y,va,"flat" not in name)
        admitted = eligible(p,cue,sem,t,k)
        metrics[name] = {"threshold":t,"support":k}
        for split in ("validation","test","external"):
            ix = np.flatnonzero((sp==split)&strict)
            metrics[name][split] = {**summarize(y[ix],p[ix]),**measure_admissions(y[ix],admitted[ix],group[ix])}
            predictions.extend([name,split,group[i],rows[i]["comment"],int(y[i]),float(p[i]),bool(admitted[i])] for i in ix)
    (out/"issue-results.json").write_text(json.dumps(metrics))
    (out/"issue-predictions.json").write_text(json.dumps(predictions,separators=(",",":")))
    (out/"metrics.json").write_text(json.dumps({"ap":metrics["separate_expert"]["test"]["ap"]}))


if __name__ == "__main__":
    run(int(os.environ["TRAINLAB_SEED"]),Path(os.environ["TRAINLAB_OUTPUT_DIR"]))
