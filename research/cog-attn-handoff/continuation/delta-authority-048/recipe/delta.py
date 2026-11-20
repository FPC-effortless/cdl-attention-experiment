"""Candidate-wise differential verification for a public dialogue state slot."""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from state_store import World

ROOT=Path(__file__).resolve().parent
AREAS=("","centre","east","north","south","west","dontcare")
LABELS=("hold",*AREAS)
SLOT_PATTERNS={
    "centre":r"\b(?:centre|center|central|downtown)\b",
    "north":r"\b(?:north|northern)\b",
    "south":r"\b(?:south|southern)\b",
    "east":r"\b(?:east|eastern)\b",
    "west":r"\b(?:west|western)\b",
    "dontcare":r"\b(?:any area|either area|doesn.t matter|no preference|don.t care|anywhere|any part)\b",
    "":r"\b(?:remove|clear|forget|drop)\b",
}
MARKED={k:re.compile(v,re.I) for k,v in SLOT_PATTERNS.items()}
HOTEL=re.compile(r"\b(?:hotel|guesthouse|guest house|accommodation|lodge|place to stay)\b",re.I)
AFFIRM=re.compile(r"^(?:yes|yeah|yep|that(?:'s| is) fine|sounds good|please do)\b",re.I)


def masked(sentence,area):
    text=sentence
    for value,pattern in MARKED.items():
        text=pattern.sub(" TARGET " if value==area else " OTHER ",text)
    return text


def document(row,area=None):
    if area is None:
        return f"SYSTEM {row['system']} USER {row['user']}"
    utterance=masked(row["user"],area)
    system=masked(row["system"],area)
    support="user_target" if MARKED[area].search(row["user"]) else "no_user_target"
    domain="user_hotel" if HOTEL.search(row["user"]) else "no_user_hotel"
    return f"PROPOSE {area or 'clear area'} {support} {domain} SYSTEM {system} USER {utterance}"


def visible_support(row,area):
    user,system=row["user"].strip(),row["system"]
    return bool(MARKED[area].search(user)) or bool(AFFIRM.match(user) and HOTEL.search(system) and MARKED[area].search(system))


class Features:
    def __init__(self):
        self.words=TfidfVectorizer(ngram_range=(1,3),min_df=3,max_features=40000,sublinear_tf=True)
        self.chars=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,4),min_df=4,max_features=30000,sublinear_tf=True)

    def fit(self,values):
        return hstack((self.words.fit_transform(values),self.chars.fit_transform(values))).tocsr()

    def transform(self,values):
        return hstack((self.words.transform(values),self.chars.transform(values))).tocsr()


def fit_models(train):
    base=Features()
    model=LogisticRegression(max_iter=250,solver="lbfgs",class_weight="balanced",C=3.)
    model.fit(base.fit([document(r) for r in train]),[r["delta"] for r in train])
    binary=Features()
    docs,targets=[],[]
    rng=np.random.default_rng(1107)
    for r in train:
        active={k for k in AREAS if visible_support(r,k)}
        active.update(AREAS[i] for i in rng.choice(len(AREAS),2,replace=False))
        if r["delta"]!="hold":
            active.add(r["delta"])
        for area in sorted(active):
            docs.append(document(r,area))
            targets.append(area==r["delta"])
    verifier=LogisticRegression(max_iter=250,solver="lbfgs",class_weight="balanced",C=2.)
    verifier.fit(binary.fit(docs),targets)
    return base,model,binary,verifier,len(docs)


def hypotheses(row,area):
    text=f"Assistant: {row['system']} User: {row['user']}"
    if not area:
        claim="In the latest turn the user explicitly removes the preferred area for the hotel."
    elif area=="dontcare":
        claim="In the latest turn the user tells the assistant that any hotel area is acceptable."
    else:
        claim=f"In the latest turn the user asks for the hotel to be in the {area} area."
    return text,claim


def entail(rows):
    directory=ROOT/"nli"
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    options.inter_op_num_threads=1
    runtime=ort.InferenceSession(str(directory/"model_quint8_avx2.onnx"),options,providers=["CPUExecutionProvider"])
    keys={k.name for k in runtime.get_inputs()}
    entails=json.loads((directory/"config.json").read_text())["label2id"]["entailment"]
    result=[]
    for k in range(0,len(rows),80):
        premise,hyp=zip(*rows[k:k+80],strict=True)
        ts=tok(list(premise),list(hyp),padding=True,truncation=True,max_length=200,return_tensors="np")
        logits=runtime.run(None,{name:value for name,value in ts.items() if name in keys})[0]
        e=np.exp(logits-logits.max(1,keepdims=True))
        result.extend((e[:,entails]/e.sum(1)).tolist())
    return result


def features(raw,models,use_nli=True):
    base,model,binary,verifier,_=models
    pred=model.predict_proba(base.transform([document(r) for r in raw]))
    c=[int(np.where(model.classes_==value)[0][0]) for value in AREAS]
    probs=pred[:,c]
    delta=verifier.predict_proba(binary.transform([document(r,v) for r in raw for v in AREAS]))[:,1]
    delta=delta.reshape(len(raw),len(AREAS))
    candidates=(np.sqrt(probs)*delta).argmax(1)
    data=[]
    queue=[]
    for i,(r,choice) in enumerate(zip(raw,candidates,strict=True)):
        mode=AREAS[int(choice)]
        raw_mode=model.classes_[pred[i].argmax()]
        queue.append(hypotheses(r,mode))
        data.append(dict(candidate=mode,raw=raw_mode,raw_score=float(pred[i].max()),
                         p=float(probs[i,choice]),delta=float(delta[i,choice]),
                         explicit=visible_support(r,mode),nli=None))
    if use_nli:
        for row,score in zip(data,entail(queue),strict=True):
            row["nli"]=score
    return data


def evaluate(raw,scores,mode,gate,replay=False):
    state={}
    correct=wrong=changes=match=take=0
    detail=[]
    p,threshold,nli,explicit=gate
    store=World() if replay else None
    for r,s in zip(raw,scores,strict=True):
        old=state.get(r["dialogue"],"")
        changes+=r["delta"]!="hold"
        cp=s["raw"] if mode=="raw" else s["candidate"]
        release=(s["raw_score"]>=p if mode=="raw" else
                 s["p"]>=p and s["delta"]>=threshold and (not explicit or s["explicit"])
                 and (mode!="nli" or s["nli"]>=nli))
        action=cp if release and cp!="hold" and cp!=old else "hold"
        if action!="hold":
            take+=1
            correct+=action==r["delta"]
            wrong+=action!=r["delta"]
            state[r["dialogue"]]=action
            if store is not None:
                assert store.commit(f"{r['dialogue']}:{r['turn']}",0,{r["dialogue"]:action})=="committed"
        match+=state.get(r["dialogue"],"")==r["state"]
        if replay:
            detail.append(dict(shard=r["shard"],dialogue=r["dialogue"],turn=r["turn"],action=action,
                               value=state.get(r["dialogue"],""),truth=r["delta"],annotated=r["state"]))
    if store is not None:
        assert store.state()==state
        store.db.close()
    return dict(count=len(raw),updates=changes,correct=correct,wrong=wrong,admitted=take,
                precision=correct/take if take else None,recall=correct/changes,state_match=match,details=detail)


def choose(raw,scores,mode):
    if mode=="raw":
        gates=[(p,0.,0.,False) for p in (.5,.65,.8,.9,.95,.975,.99,.995,1.01)]
    else:
        gates=[(p,d,n,e) for p in (.05,.25,.50) for d in (.6,.8,.9,.95,.975,.99,1.01)
              for n in ((.3,.55,.8,.95) if mode=="nli" else (0.,)) for e in (False,True)]
    valid=[]
    for gate in gates:
        v=evaluate(raw,scores,mode,gate)
        if v["wrong"]*100<=v["admitted"]:
            valid.append((v["correct"],v["admitted"],gate))
    return max(valid)[2]


def run():
    raw=json.loads((ROOT/"turns.json").read_text())
    start=time.perf_counter_ns()
    models=fit_models(raw["train"])
    fit_ns=time.perf_counter_ns()-start
    score={k:features(raw[k],models) for k in ("dev","test")}
    results={}
    for mode in ("raw","delta","nli"):
        gate=choose(raw["dev"],score["dev"],mode)
        results[mode]=dict(gate=gate,calibration=evaluate(raw["dev"],score["dev"],mode,gate),
                           test=evaluate(raw["test"],score["test"],mode,gate,True),
                           unfiltered=evaluate(raw["test"],score["test"],mode,(0,0,0,False)))
        print(mode,{k:v for k,v in results[mode]["test"].items() if k!="details"},flush=True)
    return dict(train=len(raw["train"]),verified_training_pairs=models[-1],train_ns=fit_ns,
                scores=score,results=results)


if __name__=="__main__":
    out=Path(os.environ.get("TRAINLAB_OUTPUT_DIR","."))
    (out/"delta.json").write_text(json.dumps(run(),separators=(",",":")))
