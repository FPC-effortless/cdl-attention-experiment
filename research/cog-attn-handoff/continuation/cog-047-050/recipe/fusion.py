"""Frozen score fusion and complete-utterance structural admission."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from binding import KNOWN, words_for_seed
from language import NAMES
from transfer import ContextHead, contextual, load_encoder
from nli_transfer import Entailment, CAL_CLEAR, CAL_AMBIG
from paraphrases import generate

ROOT=Path(__file__).resolve().parent
PAIR=[(i,j) for i in range(3) for j in range(3) if i!=j]
ACTIONS=[(op,s,d) for op in range(4) for s,d in PAIR]
SEEDS=(677,701,733)
EXPLICIT=(
    ("use {v} from {s} into {d} ; leave {u} alone", "perform {v} with source {s} and destination {d} ; preserve {u}"),
    ("apply {v} to {d} from {s} ; do not modify {u}", "keep {u} unchanged ; {v} from {s} into {d}"),
    ("from {s} into {d} apply {v} ; ignore {u}", "for {v} , source is {s} and destination is {d} ; leave {u} alone"),
)
IMPLICIT=(
    ("{s} has the operand . eventually {d} should show the {v} result , with no change to {u}",
     "make the balance in {d} reflect {v} on the entry for {s} . disregard {u}"),
    ("the input to {v} can be found under {s} . the tally in {d} is to change . {u} is outside this edit",
     "{u} is already right . {d} should be the result of {v} using the reference figure under {s}"),
    ("what {s} holds is the basis for the {v} calculation . post the result where the sheet says {d} . the line marked {u} is fine",
     "{d} belongs at the end of this {v} . its input is listed alongside {s} rather than the other sheet , {u}",
     "leave {u} as written . the number on file with {s} controls {v} ; move the outcome over to {d}"),
)
CORRECTION=(
    ("earlier we asked for {u} . use {s} as the input for {v} instead and put the result in {d}",),
    ("the receiving account was listed as {u} . change it to {d} ; keep {s} as the {v} operand",),
    ("ignore the old draft naming {u} as the source . {s} provides the figure for {v} and {d} should get it",
     "use {s} for {v} . the answer will go to {d} now , even though we initially proposed {u}"),
)
AMBIG=(
    ("the input for {v} in {d} could be {s} or {u} ; there is no choice yet",),
    ("run {v} on {s} when we settle the destination , perhaps {d} and perhaps {u}",),
    ("for {v} , source is {s} and destination is {d} ; leave {u} alone . wait , that destination might actually be {u}",
     "{s} or {u} would give a different result under {v} . put the result in {d} after we choose"),
)
PATTERNS={"typed":EXPLICIT,"implicit":IMPLICIT,"correction":CORRECTION,"ambiguous":AMBIG}


def ordered_tokens(tokens, names, op_map):
    # Declarative contract covering six disclosed phrasings. Literal anchors
    # cover the whole instruction, including negative and trailing text.
    for spec in sum(EXPLICIT,()):
        pat=re.escape(spec)
        for field in ("s","d","u","v"):
            pat=pat.replace("\\{"+field+"\\}",f"(?P<{field}>[a-z]+)")
        match=re.fullmatch(pat," ".join(tokens))
        if match:
            fields=match.groupdict()
            if sorted(fields[k] for k in ("s","d","u"))!=sorted(names) or fields["v"] not in op_map:
                continue
            return (op_map[fields["v"]],names.index(fields["s"]),names.index(fields["d"]))
    return None


def vocab_lexicon(lexicon,words):
    return {w:int(np.argmax(lexicon[w])) for w in words if max(lexicon[w])>=.99}


def data_for(seed,split):
    extra=words_for_seed(seed)
    return {k:generate(seed+1000+split*101+i,64 if split<2 else 128,extra,f[split])
            for i,(k,f) in enumerate(PATTERNS.items())}


def original_nli(seed):
    p=ROOT/f"nli-lexicon-{seed}.json"
    return json.loads(p.read_text())


def embeddings(raw,extra,nli,encoder,head):
    verbs=KNOWN+extra
    with torch.no_grad():
        nr,_=nli.roles_and_clarity(raw,verbs,"separate")
        feats=contextual(raw,verbs,encoder)
        mr=head.role(feats).softmax(-1)
        clear=head.clear(feats).sigmoid()
    return nr.numpy(),mr.numpy(),clear.numpy()


def rows_for(raw,group,seed,split,nli,encoder,head):
    extra=words_for_seed(seed)
    lex=original_nli(seed)
    op_map=vocab_lexicon(lex,extra)
    nr,mr,clarity=embeddings(raw,extra,nli,encoder,head)
    out=[]
    for r,n,m,c in zip(raw,nr,mr,clarity,strict=True):
        tokens,names,state,target,*truth=r
        verb=next(w for w in tokens if w in extra)
        known=lex[verb]
        op=int(np.argmax(known))
        contract=ordered_tokens(tokens,names,op_map)
        a=np.argsort(n);b=np.argsort(m)
        feats=[float(n[-1]),float(n[a[-1]]),float(n[a[-2]]),float(m[b[-1]]),float(m[b[-2]]),float(c),
               float(a[-1]==b[-1]),len(tokens)/40,max(known),float(contract is not None)]
        best=(op,*PAIR[int((np.log(n+1e-10)+np.log(m+1e-10)).argmax())])
        out.append(dict(tokens=tokens,names=names,state=state.tolist(),label=None if group=="ambiguous" else truth,
                        nli=[op,*PAIR[int(a[-1])]],local_contract=None if contract is None else list(contract),
                        blended=list(best),features=feats,split=split,group=group,seed=seed))
    return out


def counts(rows,selector):
    ret={}
    for x in rows:
        r=ret.setdefault(x["group"],[0,0,0])
        r[0]+=1
        y=selector(x)
        if y is not None:
            r[1]+=1;r[2]+=y==x["label"]
    return ret


def run(out):
    nli=Entailment()
    encoder=load_encoder("pretrained")
    data=[]
    for i,seed in enumerate(SEEDS):
        head=ContextHead()
        with np.load(ROOT/f"minilm-head-{i}.npz") as f:
            head.load_state_dict({k:torch.tensor(f[k]) for k in f.files})
        head.eval()
        for split in range(3):
            for group,rows in data_for(seed,split).items():
                data.extend(rows_for(rows,group,seed,split,nli,encoder,head))
        print(seed,"scored",flush=True)
    train=[r for r in data if r["split"]==0];cal=[r for r in data if r["split"]==1];test=[r for r in data if r["split"]==2]
    classifier=LogisticRegression(max_iter=1000).fit([r["features"] for r in train],[r["blended"]==r["label"] for r in train])
    for r,p in zip(data,classifier.predict_proba([r["features"] for r in data])[:,1],strict=True):
        r["risk_score"]=float(p)
    def proposed(r,threshold):
        if r["local_contract"] is not None:
            return r["local_contract"]
        return r["blended"] if r["risk_score"]>=threshold else None
    choices=[]
    for t in (.8,.9,.95,.97,.98,.99,.995,.999,1.01):
        c=counts(cal,lambda r:proposed(r,t))
        valid=[v for k,v in c.items() if k!="ambiguous"]
        n=sum(v[1] for v in valid);ok=sum(v[2] for v in valid)
        if n and ok/n>=.99 and c["ambiguous"][1]<=.01*c["ambiguous"][0]:
            choices.append((n,t))
    threshold=max(choices,default=(0,1.01))[1]
    report=dict(threshold=threshold,calibration=counts(cal,lambda r:proposed(r,threshold)),
                results=dict(structural=counts(test,lambda r:r["local_contract"]),
                nli_raw=counts(test,lambda r:r["nli"]),blend_raw=counts(test,lambda r:r["blended"]),
                calibrated=counts(test,lambda r:proposed(r,threshold))),rows=data)
    Path(out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report["results"],indent=2),flush=True)


if __name__=="__main__":
    run(sys.argv[1])
