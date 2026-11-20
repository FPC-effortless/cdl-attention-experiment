"""Learn local governing-mention scores and reload the auditable model."""
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import bridge

reader=bridge.reader
BOUNDARY=-1
FAMILY_FEATURES=(
    "choose {p} instead of {q}", "skip {q} and choose {p}",
    "select {p} then ignore {q}", "over {q} use {p}", "{p} rather than {q}",
    "discard {q} and keep {p}", "ignore {q} but select {p}", "prefer {p} over {q}",
    "retain {p} and discard {q}", "drop {q} select {p}", "keep {p} not {q}",
)
for word in ("of","skip","then","drop","not","instead"):
    reader.V.setdefault(word,len(reader.V)+1)


def features(tokens,position):
    # No identity index or global entity rank enters the selector.
    ids={reader.V[n] for n in reader.WORDS}
    values=[tok if tok not in ids else BOUNDARY for tok in tokens]
    out={}
    for j in range(-2,3):
        if j==0:
            continue
        i=position+j
        if 0<=i<len(values):
            out[f"offset:{j}:{values[i]}"]=1.
    for shift in (-2,1):
        i=position+shift
        if i>=0 and i+1<len(values):
            out[f"pair:{shift}:{values[i]}:{values[i+1]}"]=1.
    return out


class Scope:
    def __init__(self,names,coef,bias):
        self.weights=dict(zip(names,coef,strict=True))
        self.bias=bias

    def scores(self,query):
        register={reader.V[v] for v in reader.WORDS}
        choices=[]
        for i,token in enumerate(query):
            if token in register:
                value=self.bias+sum(self.weights.get(k,0.)*v for k,v in features(query,i).items())
                choices.append((float(value),int(token)))
        return sorted(choices,reverse=True)

    def pick(self,documents,query,model):
        cues=[]
        for reads in query:
            selected=self.scores(reads[0])
            # The held suite declares two entity mentions and a supported local cue.
            cues.append([reader.V["entry"],selected[0][1],0,0,0,0] if len(selected)==2 and
                        selected[0][0]-selected[1][0]>.05 else [0]*6)
        return bridge.csm_pick(model,documents,np.array(cues)[:,None,:])[:,0],np.array(cues)[:,1]!=0


def fit(seed):
    documents,questions,truth=bridge.csm_data(seed+883,1000,FAMILY_FEATURES)
    ids={reader.V[n] for n in reader.WORDS}
    rows,correct=[],[]
    for memory,paired,targets in zip(documents,questions,truth,strict=True):
        for text,slot in zip(paired,targets,strict=True):
            relevant=set(memory[2*slot:2*slot+2].reshape(-1))&ids
            for i,token in enumerate(text):
                if token in ids:
                    rows.append(features(text,i))
                    correct.append(token in relevant)
    vec=DictVectorizer()
    model=LogisticRegression(C=3.,max_iter=200).fit(vec.fit_transform(rows),correct)
    names=vec.get_feature_names_out()
    coefs=model.coef_[0]
    bias=float(model.intercept_[0])
    return Scope(names,coefs,bias),dict(names=names,coef=coefs,intercept=bias)
