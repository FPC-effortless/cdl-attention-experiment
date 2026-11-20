"""An independently fitted sparse model on public dialogue text and official splits."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

from state_store import World

ROOT=Path(__file__).resolve().parent


def select(probs,classes,threshold):
    result=[]
    for row in probs:
        idx=int(row.argmax())
        result.append(classes[idx] if row[idx]>=threshold else "hold")
    return result


def oracle(raw,predicted):
    correct=pushed=0
    rows=[]
    world=World()
    for datum,action in zip(raw,predicted,strict=True):
        session=datum["dialogue"]
        if action!="hold":
            world.commit(f"{session}:{datum['turn']}",0,{session:action})
        current=world.db.execute("SELECT value FROM state WHERE k=?",(session,)).fetchone()
        value=json.loads(current[0]) if current else ""
        correct+=value==datum["target"]
        pushed+=action!="hold" and action==datum["action"]
        rows.append(dict(dialogue=session,turn=datum["turn"],action=action,state=value))
    world.db.close()
    updates=sum(a!="hold" for a in predicted)
    truth=sum(d["action"]!="hold" for d in raw)
    return dict(update_attempts=updates,correct_updates=pushed,gold_updates=truth,state_matches=correct,
                count=len(raw),precision=pushed/updates if updates else None,recall=pushed/truth if truth else 0,rows=rows)


def run():
    data=json.loads((ROOT/"dialogue-expanded.json").read_text())
    results={}
    for mode,column in (("current","current"),("persistent","context")):
        words=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=30000,sublinear_tf=True)
        chars=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),min_df=3,max_features=50000,sublinear_tf=True)
        train=[d[column] for d in data["train"]]
        start=time.perf_counter_ns()
        x=hstack((words.fit_transform(train),chars.fit_transform(train))).tocsr()
        model=LogisticRegression(max_iter=240,class_weight="balanced",solver="lbfgs",C=3.)
        model.fit(x,[d["action"] for d in data["train"]])
        fit=time.perf_counter_ns()-start
        classes=model.classes_.tolist()
        predictions={}
        for split in ("dev","test"):
            text=[d[column] for d in data[split]]
            x=hstack((words.transform(text),chars.transform(text))).tocsr()
            predictions[split]=model.predict_proba(x)
        candidates=[]
        for threshold in (.50,.65,.75,.80,.85,.90,.95,.98,.99,1.01):
            check=oracle(data["dev"],select(predictions["dev"],classes,threshold))
            if check["precision"] is None or check["precision"]>=.99:
                candidates.append((check["correct_updates"],threshold))
        threshold=max(candidates)[1]
        results[mode]=dict(fit_ns=fit,train_n=len(train),threshold=threshold,
                            guarded=oracle(data["test"],select(predictions["test"],classes,threshold)),
                            raw=oracle(data["test"],select(predictions["test"],classes,0.)),
                            raw_scores=predictions["test"].tolist(),classes=classes)
    return results
