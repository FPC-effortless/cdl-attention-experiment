"""Use a pinned inference model to score explicit operand and destination hypotheses."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from binding import KNOWN, lexical_ids, rows, words_for_seed
from language import NAMES, encode
from paraphrases import CAL_AMBIG,CAL_CLEAR,generate
from transfer import calibrate

ROOT = Path(__file__).resolve().parent
PAIR = [(i,j) for i in range(3) for j in range(3) if i!=j]
HELD = {
    "implicit": (
        "Here's how to {v} this stock entry: the number kept on {s} determines it, and afterwards {d} has the new total. Leave {u} as it was.",
        "{u} is outside the scope of this job. The {v} output goes on {d}'s line, driven by the tally you see at {s}.",
        "Change the line marked {d} after running {v}. What you need as the operand is the figure for {s}; {u} is a separate tally.",
        "{d} should show the result at the end, and {u} should look the same. Use the number under {s} when you {v}.",
    ),
    "correction": (
        "{u} is written on my earlier note as the source. That is incorrect. Use {s} for the {v} adjustment in {d}.",
        "Please {v} the quantity in {s}. I had named {u} to receive it, but the final choice is {d}.",
        "There was a question about {u} or {s} supplying the value. It is settled: use {s} and enter the {v} result in {d}.",
        "The earlier line for {u} was a draft. Replace it in the instructions with {d} as the destination for {v}; still take the number from {s}.",
    ),
    "ambiguous": (
        "The {v} result belongs on the line for {d}, but {s} and {u} have different possible inputs. Which of them should we use?",
        "Run {v} with the figure from {s}; both {d} and {u} are listed for the result, and I haven't checked which is correct.",
        "A note from yesterday says the source is {s}. Another note says {u}. I haven't settled the source for this {v} change to {d}.",
        "{s} provides the value for {v}. If the approved sheet is {d}, put it there; otherwise use {u}. The approval has not come through.",
    ),
}


def fixture(seed):
    extra = words_for_seed(seed)
    cal = {"clear":generate(seed+307,128,extra,CAL_CLEAR),
           "ambiguous":generate(seed+311,128,extra,CAL_AMBIG)}
    held = {key:generate(seed+330+i,128,extra,patterns) for i,(key,patterns) in enumerate(HELD.items())}
    return extra,rows(seed+305,extra,96,"support"),cal,held


class Entailment:
    def __init__(self):
        models=ROOT/"nli"
        self.tokenizer=AutoTokenizer.from_pretrained(models,local_files_only=True,use_fast=True)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(models/"model_quint8_avx2.onnx"),options,
                                          providers=["CPUExecutionProvider"])
        self.inputs={item.name for item in self.session.get_inputs()}
        labels=json.loads((models/"config.json").read_text())["label2id"]
        self.entailment=int(labels["entailment"])

    def roles_and_clarity(self,raw,operands,mode):
        all_input,all_labels=[],[]
        for tokens,names,*_ in raw:
            premise=" ".join("adjust" if t in operands else t for t in tokens)
            if mode=="pair":
                hypotheses=[f"The input amount comes from {names[src]} and the only stock balance to update is {names[dst]}."
                            for src,dst in PAIR]
            else:
                hypotheses=([f"The input amount for the calculation comes from {name}." for name in names]+
                            [f"The result of the calculation is recorded in {name}." for name in names])
            hypotheses.append("The instruction leaves it unresolved which account is the input or which account is the destination.")
            all_input.extend([premise]*len(hypotheses))
            all_labels.extend(hypotheses)
        logits=[]
        for i in range(0,len(all_input),96):
            encoded=self.tokenizer(all_input[i:i+96],all_labels[i:i+96],padding=True,return_tensors="np")
            model_in={k:v for k,v in encoded.items() if k in self.inputs}
            logits.append(self.session.run(None,model_in)[0])
        entailment=torch.tensor(np.concatenate(logits)).softmax(-1)[:,self.entailment].reshape(-1,7)
        evidence=entailment[:,:6].clamp_min(1e-8).log()
        if mode=="separate":
            evidence=torch.stack([evidence[:,s]+evidence[:,3+d] for s,d in PAIR],dim=1)
        return evidence.softmax(1),1-entailment[:,6]


def run(seed,arm):
    extra,support,cal,held=fixture(seed)
    operands=KNOWN+extra
    vocabulary={w:i+4 for i,w in enumerate(sorted({t for row in support for t in row[0] if t not in NAMES}))}
    names=list(range(len(vocabulary)+4,len(vocabulary)+7))
    model=Entailment()
    mode="pair" if arm=="candidate" else "separate"
    role,_=model.roles_and_clarity(support,operands,mode)
    _,possible,_,_=encode(support,vocabulary,names)
    word_ids=lexical_ids(support,vocabulary,extra)
    likelihood=(.98*(role[:,None,:]*possible.reshape(-1,4,6)).sum(-1)+.02/512).log()
    lexicon={vocabulary[w]:likelihood[word_ids==vocabulary[w]].sum(0).softmax(-1) for w in extra}

    def pred(raw,labeled):
        _,compatible,gold,_=encode(raw,vocabulary,names)
        role,clarity=model.roles_and_clarity(raw,operands,mode)
        operations=torch.stack([lexicon[int(i)] for i in lexical_ids(raw,vocabulary,extra)])
        joint=(operations[:,:,None]*role[:,None,:]).flatten(1)
        confidence,choice=joint.max(1)
        return dict(pred=choice.tolist(),confidence=confidence.tolist(),clear=clarity.tolist(),posterior=joint.tolist(),
                    correct=(choice==gold).tolist() if labeled else None,
                    effect=compatible[torch.arange(len(choice)),choice].tolist() if labeled else None)

    calibration={k:pred(raw,k!="ambiguous") for k,raw in cal.items()}
    return [dict(mode=mode,calibration=calibration,gate=calibrate(calibration["clear"],calibration["ambiguous"]),
                 rows={k:pred(raw,k!="ambiguous") for k,raw in held.items()},
                 lexicon={w:lexicon[vocabulary[w]].tolist() for w in extra})]
